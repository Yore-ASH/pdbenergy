"""Tests for the web GUI: job control, introspection, and the HTTP surface.

The GUI is a thin layer over the CLI, so these tests focus on the things that
can actually break without anyone noticing:

* a job's exit status and output reach the client, and Cancel really kills it;
* leaving a job running does not block the server (threading);
* the read-only introspection degrades gracefully on an empty project;
* the static-file route cannot be walked out of its directory;
* a config change (the residue caps) re-filters an existing inventory rather
  than trusting the stale ``is_usable`` flag baked into it.

The HTTP tests run a real server on a background thread, because the interesting
failures here are in the routing and streaming, not in the handler functions.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.config import Config                              # noqa: E402
from pdbenergy.gui import (                                      # noqa: E402
    Handler,
    JobManager,
    _usable_at,
    list_pdb_files,
    list_runs,
    predict_checkpoints,
    project_root,
    project_state,
    resolve_dir,
)

SLEEPER = "import time, sys\nprint('started', flush=True)\nfor i in range(120):\n" \
          "    print('tick', i, flush=True)\n    time.sleep(0.25)\n"

#: What ``python -m pdbenergy.cli`` prints when the package cannot be found.
#: The message names the *dotted* module even when the parent is what is
#: missing, which is why it used to be so hard to act on.
MISSING_MODULE_MESSAGE = "No module named pdbenergy.cli"


class TestJobManager(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: every job runs with this directory as its cwd, so
        # on Windows the removal can race a closing process handle and fail with
        # WinError 32.  The assertions have already run by then; failing the whole
        # suite over a leftover directory in %TEMP% would be noise.
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = self.tmp.name
        self.manager = JobManager(self.dir)

    def tearDown(self):
        for job in list(self.manager._jobs.values()):
            if job.process and job.status == "running":
                job.process.kill()
        self.tmp.cleanup()

    def _wait(self, job_id, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.manager.get(job_id)
            if job.status != "running":
                return job
            time.sleep(0.05)
        self.fail(f"job {job_id} did not finish within {timeout}s")

    def test_job_captures_output_and_exit_code(self):
        job = self.manager.start("help", ["--help"])
        done = self._wait(job.id)
        self.assertEqual(done.status, "done")
        self.assertEqual(done.returncode, 0)
        text = "\n".join(done.lines)
        self.assertIn("usage", text.lower())

    def test_failing_job_is_marked_failed(self):
        # An unknown subcommand exits non-zero; the GUI must surface that.
        job = self.manager.start("bad", ["definitely-not-a-command"])
        done = self._wait(job.id)
        self.assertEqual(done.status, "failed")
        self.assertNotEqual(done.returncode, 0)

    def test_missing_module_is_failed_even_when_the_child_exits_zero(self):
        """A child that could not import anything must never look successful.

        This pins down the bug that produced a "✓ 任务完成" for a run that
        labelled nothing: the child printed ``No module named pdbenergy.cli``
        and still exited 0, so a return-code-only check called it a success.
        """
        script = os.path.join(self.dir, "liar.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(f"print({MISSING_MODULE_MESSAGE!r})\n")
        job = self.manager.start("liar", [], module=script)
        done = self._wait(job.id)
        self.assertEqual(done.returncode, 0, "the child really does exit 0")
        self.assertEqual(done.status, "failed")
        text = "\n".join(done.lines)
        # The log must explain itself: which interpreter, which project root,
        # and what to run to check by hand.
        self.assertIn(self.manager.project_root, text)
        self.assertIn("pdbenergy.cli", text)
        self.assertIn("-c", text)

    def test_real_missing_module_is_failed_and_explained(self):
        """End to end: ask for a module that does not exist, anywhere."""
        job = self.manager.start("nope", ["--help"], module="pdbenergy.no_such_module")
        done = self._wait(job.id)
        self.assertEqual(done.status, "failed")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no_such_module", "\n".join(done.lines))

    def test_manager_falls_back_to_the_project_root(self):
        # Started from a shortcut, ``os.getcwd()`` can be anywhere - even a
        # directory that no longer exists.  Jobs must still find data/.
        self.assertEqual(JobManager().cwd, project_root())

    def test_missing_cwd_does_not_break_launching(self):
        missing = os.path.join(self.dir, "was-renamed-away")
        manager = JobManager(missing)
        self.assertEqual(manager.cwd, project_root())
        job = manager.start("help", ["--help"])
        self.assertEqual(self._wait_for(manager, job.id).status, "done")
        manager.shutdown()

    def test_child_env_puts_the_project_root_first_on_pythonpath(self):
        manager = JobManager(self.dir)
        parts = manager.child_env()["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(parts[0], manager.project_root)

    def test_child_env_strips_interpreter_poison(self):
        """An inherited PYTHONHOME makes the venv interpreter use another
        installation's stdlib.  Every path the GUI prints still looks correct,
        which is what made this so hard to see from a failing job log."""
        manager = JobManager(self.dir)
        with mock.patch.dict(os.environ, {"PYTHONHOME": r"C:\Python313",
                                          "PYTHONSAFEPATH": "1",
                                          "PYTHONSTARTUP": "x.py"}, clear=False):
            env = manager.child_env()
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONSAFEPATH", env)
        self.assertNotIn("PYTHONSTARTUP", env)
        self.assertEqual(sorted(manager.dropped_env_keys),
                         ["PYTHONHOME", "PYTHONSAFEPATH", "PYTHONSTARTUP"])
        # ...and the log says so, instead of dropping them silently.
        report = "\n".join(manager.python_env_report())
        self.assertIn("PYTHONHOME", report)

    def test_diagnose_child_reports_the_working_sys_path(self):
        """The probe must actually run and describe a healthy child."""
        manager = JobManager(self.dir)
        text = "\n".join(manager.diagnose_child("pdbenergy.cli"))
        self.assertIn("executable", text)
        self.assertIn("sys.path", text)
        self.assertIn("find_spec", text)
        self.assertIn("import     : ok", text)
        self.assertIn(manager.project_root, text)

    def test_child_env_is_clean_even_when_the_gui_is_poisoned(self):
        """The regression, end to end: poison the manager, not the process."""
        manager = JobManager(self.dir)
        manager.project_root = project_root()
        with mock.patch.dict(os.environ, {"PYTHONHOME": r"C:\Python313"}, clear=False):
            job = manager.start("help", ["--help"])
            done = self._wait_for(manager, job.id)
        self.assertEqual(done.status, "done",
                         "a poisoned parent environment must not reach the child")
        manager.shutdown()

    @staticmethod
    def _wait_for(manager, job_id, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = manager.get(job_id)
            if job.status != "running":
                return job
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} did not finish within {timeout}s")

    def test_cancel_terminates_a_running_job(self):
        script = os.path.join(self.dir, "sleeper.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(SLEEPER)
        job = self.manager.start("sleep", [], module="sleeper.py")
        time.sleep(1.2)                       # let it start producing output
        self.assertEqual(self.manager.get(job.id).status, "running")
        self.assertTrue(self.manager.cancel(job.id))
        done = self._wait(job.id, timeout=20)
        self.assertEqual(done.status, "cancelled")
        self.assertIn("cancelled by user", "\n".join(done.lines))
        self.assertGreater(done.produced, 1)

    def test_cancel_is_a_no_op_after_completion(self):
        job = self.manager.start("help", ["--help"])
        self._wait(job.id)
        self.assertFalse(self.manager.cancel(job.id))

    def test_public_cursor_streams_only_new_lines(self):
        job = self.manager.start("help", ["--help"])
        done = self._wait(job.id)
        first = done.public(0)
        self.assertTrue(first["lines"])
        again = done.public(first["cursor"])
        self.assertEqual(again["lines"], [])

    def test_long_job_does_not_block_the_manager(self):
        script = os.path.join(self.dir, "sleeper.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(SLEEPER)
        slow = self.manager.start("slow", [], module="sleeper.py")
        try:
            t0 = time.time()
            quick = self.manager.start("quick", ["--help"])
            self.assertLess(time.time() - t0, 5.0, "starting a job must not block")
            self._wait(quick.id, timeout=30)
        finally:
            self.manager.cancel(slow.id)
        self.assertEqual(self.manager.get(slow.id).status, "cancelled")


    def test_shutdown_terminates_running_jobs(self):
        """Closing the GUI must not leave OpenMM jobs running for hours.

        Terminating the parent does not cascade to ``subprocess.Popen`` children,
        so the manager has to do it explicitly, and this test is what stops that
        from silently regressing.
        """
        script = os.path.join(self.dir, "sleeper.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(SLEEPER)
        jobs = [self.manager.start(f"sleep{i}", [], module="sleeper.py") for i in range(3)]
        time.sleep(1.0)
        self.assertEqual(len(self.manager.running()), 3)
        stopped = self.manager.shutdown()
        self.assertEqual(stopped, 3)
        self.assertEqual(self.manager.running(), [])
        for job in jobs:
            self.assertEqual(job.status, "cancelled")
            self.assertIsNotNone(job.process.poll(), "child process still alive")

    def test_shutdown_is_safe_with_nothing_running(self):
        self.assertEqual(self.manager.shutdown(), 0)


class TestDirectoryResolution(unittest.TestCase):
    """The GUI's directory arguments are relative by default, so they must be
    anchored to the project - not to whatever cwd the launcher happened to use.

    Getting this wrong is silent: the interface simply reports an empty project
    (``data/raw  0 pdb files``) and jobs write their outputs somewhere else.
    """

    def test_relative_defaults_resolve_against_the_project(self):
        root = project_root()
        self.assertEqual(resolve_dir("data/raw"), os.path.join(root, "data", "raw"))
        self.assertEqual(resolve_dir("outputs"), os.path.join(root, "outputs"))
        self.assertEqual(resolve_dir("data/raw", root=root),
                         os.path.join(root, "data", "raw"))

    def test_absolute_paths_are_left_alone(self):
        other = os.path.join(tempfile.gettempdir(), "pdbenergy-outputs")
        self.assertEqual(resolve_dir(other), os.path.normpath(other))

    def test_a_root_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_dir("data/raw", root=tmp),
                             os.path.join(tmp, "data", "raw"))


class TestUsableAt(unittest.TestCase):
    """``is_usable`` in inventory.json is baked in at scan time; the caps in the
    UI must be able to widen the selection without a rescan."""

    def test_entry_blocked_only_by_the_cap_becomes_usable(self):
        row = {"n_residues": 300, "is_usable": False,
               "reason": "300 residues exceeds max_residues=120"}
        self.assertFalse(_usable_at(row, 120, 10))
        self.assertTrue(_usable_at(row, 300, 10))

    def test_non_protein_entry_is_never_usable(self):
        row = {"n_residues": 0, "is_usable": False,
               "reason": "no standard amino-acid residues"}
        self.assertFalse(_usable_at(row, 5000, 1))

    def test_multi_chain_entry_is_never_usable(self):
        row = {"n_residues": 80, "is_usable": False,
               "reason": "6 chains (multi-chain assemblies unsupported)"}
        self.assertFalse(_usable_at(row, 5000, 1))

    def test_parse_error_is_never_usable(self):
        row = {"n_residues": 0, "is_usable": False, "reason": "parse error: boom"}
        self.assertFalse(_usable_at(row, 5000, 1))

    def test_too_short_is_usable_only_if_min_is_lowered(self):
        row = {"n_residues": 4, "is_usable": False,
               "reason": "4 residues is below min_residues=10"}
        self.assertFalse(_usable_at(row, 120, 10))
        self.assertTrue(_usable_at(row, 120, 1))
        self.assertTrue(_usable_at(row, 120, 0))


class TestIntrospection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.raw = os.path.join(self.root, "raw")
        self.interim = os.path.join(self.root, "interim")
        self.outputs = os.path.join(self.root, "outputs")
        for d in (self.raw, self.interim, self.outputs):
            os.makedirs(d, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _state(self):
        return project_state(
            Config(), interim_dir=self.interim, raw_dir=self.raw,
            outputs_dir=self.outputs, processed_dir=os.path.join(self.root, "proc"),
        )

    def test_empty_project_reports_zeros_and_no_inventory(self):
        state = self._state()
        self.assertEqual(state["counts"], {"raw_pdb": 0, "labelled": 0, "pending": 0})
        self.assertIsNone(state["inventory"])
        self.assertEqual(state["runs"], [])
        # Defaults must still be present so the form has something to show.
        self.assertIn("n_rbf", state["defaults"]["features"])

    def test_counts_and_pending(self):
        for name in ("a.pdb", "b.pdb", "c.pdb"):
            open(os.path.join(self.raw, name), "w").close()
        open(os.path.join(self.interim, "a.npz"), "w").close()
        state = self._state()
        self.assertEqual(state["counts"]["raw_pdb"], 3)
        self.assertEqual(state["counts"]["labelled"], 1)
        self.assertEqual(state["counts"]["pending"], 2)

    def test_inventory_summary_is_read_when_present(self):
        rows = [
            {"pdb_id": "A", "is_usable": True, "n_residues": 50},
            {"pdb_id": "B", "is_usable": False, "n_residues": 900,
             "reason": "900 residues exceeds max_residues=120"},
        ]
        with open(os.path.join(self.outputs, "inventory.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        inv = self._state()["inventory"]
        self.assertEqual(inv["scanned"], 2)
        self.assertEqual(inv["usable"], 1)
        self.assertEqual(inv["unusable"], 1)
        self.assertEqual(inv["max_residues"], 50)

    def test_corrupt_inventory_does_not_crash_the_state_call(self):
        with open(os.path.join(self.outputs, "inventory.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertIsNone(self._state()["inventory"])

    def test_list_runs_needs_a_checkpoint(self):
        run = os.path.join(self.outputs, "good")
        os.makedirs(run)
        open(os.path.join(run, "checkpoint.pt"), "w").close()
        os.makedirs(os.path.join(self.outputs, "not_a_run"))
        # A directory with data files but no checkpoint is not a run.
        self.assertEqual([r["name"] for r in list_runs(self.outputs)], ["good"])
        self.assertEqual(list_runs(os.path.join(self.root, "missing")), [])

    def test_run_metrics_and_figures_are_exposed(self):
        run = os.path.join(self.outputs, "r1")
        os.makedirs(os.path.join(run, "eval"))
        open(os.path.join(run, "checkpoint.pt"), "w").close()
        with open(os.path.join(run, "data_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"n_proteins": 14, "n_samples": 730,
                       "protein_splits": {"test": ["1BDD"]}}, fh)
        with open(os.path.join(run, "eval", "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "n_parameters": 42562,
                "splits": {"test": {"overall": {"mae": 102.0, "rmse": 160.0,
                                                "r2": 0.06, "spearman": 0.43},
                                    "rank_discrimination": {"mean_spearman": 0.15}}},
                "baselines": {"test": {"mae": 107.3}},
            }, fh)
        open(os.path.join(run, "eval", "parity_test.png"), "wb").close()

        info = list_runs(self.outputs)[0]
        self.assertTrue(info["has_eval"])
        self.assertEqual(info["n_parameters"], 42562)
        self.assertAlmostEqual(info["metrics"]["test"]["mae"], 102.0)
        self.assertAlmostEqual(info["metrics"]["test"]["rank_rho"], 0.15)
        self.assertEqual(info["baselines"]["test"]["mae"], 107.3)
        self.assertIn("parity_test", info["figures"])

    def test_checkpoints_are_ordered_by_validation_mae(self):
        for name, val in (("worse", 200.0), ("best", 100.0)):
            run = os.path.join(self.outputs, name)
            os.makedirs(os.path.join(run, "eval"))
            open(os.path.join(run, "checkpoint.pt"), "w").close()
            with open(os.path.join(run, "eval", "metrics.json"), "w", encoding="utf-8") as fh:
                json.dump({"splits": {"val": {"overall": {"mae": val},
                                              "rank_discrimination": {}}}}, fh)
        names = [c["name"] for c in predict_checkpoints(self.outputs)]
        self.assertEqual(names[0], "best")

    def test_list_pdb_files_filters_and_limits(self):
        for i in range(5):
            open(os.path.join(self.raw, f"x{i}.pdb"), "w").close()
        open(os.path.join(self.raw, "notes.txt"), "w").close()
        self.assertEqual(len(list_pdb_files(self.raw)), 5)
        self.assertEqual(len(list_pdb_files(self.raw, limit=2)), 2)
        self.assertEqual(list_pdb_files(os.path.join(self.root, "nope")), [])


class TestHTTPSurface(unittest.TestCase):
    """A real server on a background thread, exercised over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = cls.tmp.name
        cls.raw = os.path.join(root, "raw")
        cls.interim = os.path.join(root, "interim")
        cls.outputs = os.path.join(root, "outputs")
        for d in (cls.raw, cls.interim, os.path.join(cls.outputs, "run1", "eval")):
            os.makedirs(d, exist_ok=True)
        open(os.path.join(cls.raw, "AAAA.pdb"), "w").close()
        open(os.path.join(cls.interim, "AAAA.npz"), "w").close()
        open(os.path.join(cls.outputs, "run1", "checkpoint.pt"), "w").close()
        with open(os.path.join(cls.outputs, "run1", "eval", "parity_test.png"), "wb") as fh:
            fh.write(b"\x89PNG")
        # A file that a traversal attempt would love to reach.
        with open(os.path.join(root, "secret.txt"), "w", encoding="utf-8") as fh:
            fh.write("do not serve me")

        handler = type("Bound", (Handler,), {
            "manager": JobManager(root),
            "cfg": Config(),
            "dirs": {"raw": cls.raw, "interim": cls.interim,
                     "processed": os.path.join(root, "proc"),
                     "outputs": cls.outputs},
        })
        cls.handler_cls = handler
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        # Terminate children *before* touching the temp directory: a job still
        # running holds a handle on its cwd, and on Windows that makes cleanup
        # fail with WinError 32.  This is also the behaviour the GUI itself needs
        # (JobManager.shutdown), so the test would rather exercise it than dodge it.
        cls.handler_cls.manager.shutdown()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path, as_json=True):
        with urllib.request.urlopen(self.url(path), timeout=30) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if as_json else body)

    def post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url(path), data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())

    def test_index_is_served(self):
        status, body = self.get("/", as_json=False)
        self.assertEqual(status, 200)
        self.assertIn(b"pdbenergy", body)
        self.assertIn(b"<title>", body)

    def test_state_endpoint(self):
        status, state = self.get("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["counts"]["raw_pdb"], 1)
        self.assertEqual(state["counts"]["labelled"], 1)
        self.assertEqual(state["counts"]["pending"], 0)

    def test_missing_inventory_is_a_clean_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/inventory")
        self.assertEqual(ctx.exception.code, 404)

    def test_jobs_listing(self):
        status, payload = self.get("/api/jobs")
        self.assertEqual(status, 200)
        self.assertIn("running", payload)
        self.assertIn("recent", payload)

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_figure_is_served(self):
        status, body = self.get("/api/figure/run1/parity_test", as_json=False)
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_figure_path_traversal_is_refused(self):
        for attempt in ("/api/figure/../../secret", "/api/figure/run1/..%2f..%2fsecret",
                        "/api/figure/run1/%2e%2e%2f%2e%2e%2fsecret"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(attempt)
            self.assertIn(ctx.exception.code, (400, 404),
                          f"{attempt} should not be served")

    def test_pdbs_and_checkpoints_endpoints(self):
        _, pdbs = self.get("/api/pdbs")
        self.assertEqual(pdbs["files"], ["AAAA.pdb"])
        _, ck = self.get("/api/checkpoints")
        self.assertEqual([c["name"] for c in ck["checkpoints"]], ["run1"])

    def test_rounds_endpoint_is_empty_before_any_run(self):
        _, payload = self.get("/api/rounds")
        self.assertEqual(payload["rounds"], [])

    def test_posting_an_unknown_action_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/job", {"action": "definitely-not-an-action"})
        self.assertEqual(ctx.exception.code, 400)

    def test_action_without_required_input_is_rejected(self):
        for payload in ({"action": "predict", "paths": [], "checkpoint": "x"},
                        {"action": "predict", "paths": ["a.pdb"]},
                        {"action": "evaluate"},
                        {"action": "label", "ids": []}):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.post("/api/job", payload)
            self.assertEqual(ctx.exception.code, 400, payload)

    def test_starting_a_job_returns_a_streamable_id(self):
        """End-to-end plumbing: POST a job, resolve its id, watch it finish.

        The action must fail *locally and fast*.  An earlier version used
        ``download`` with a bogus PDB id, which depends on network timeouts and
        retries - it passed once and then hung for minutes, which is a flaky test
        and a lesson about letting the network into a unit test.
        """
        bogus = os.path.join(self.tmp.name, "no_such_run")
        status, payload = self.post("/api/job", {"action": "evaluate", "run_dir": bogus})
        self.assertEqual(status, 200)
        job_id = payload["job"]["id"]
        self.assertTrue(job_id)
        self.assertIn("evaluate", payload["job"]["command"])

        deadline = time.time() + 90
        status_value = "running"
        while time.time() < deadline:
            _, job = self.get(f"/api/job/{job_id}")
            status_value = job["status"]
            if status_value != "running":
                break
            time.sleep(0.3)
        self.assertIn(status_value, ("done", "failed"),
                      "job never reached a terminal state")
        # A missing run directory must be reported as a failure, not a success.
        self.assertEqual(status_value, "failed")

    def test_unknown_job_id_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/job/doesnotexist")
        self.assertEqual(ctx.exception.code, 404)

    def test_server_stays_responsive_while_a_job_runs(self):
        """The whole point of threading the server: a slow job must not block it."""
        sleeper = os.path.join(self.tmp.name, "sleeper.py")
        with open(sleeper, "w", encoding="utf-8") as fh:
            fh.write(SLEEPER)
        manager = self.handler_cls.manager
        job = manager.start("slow", [], module=sleeper)
        try:
            t0 = time.time()
            for _ in range(5):
                self.get("/api/state")
            self.assertLess(time.time() - t0, 10.0,
                            "state requests were blocked by a running job")
        finally:
            manager.cancel(job.id)


class TestFrontEndConsistency(unittest.TestCase):
    """Static checks on the single-file front end.

    A mistyped element id is the classic way a panel silently dies: the script
    throws a TypeError on the first ``$('typo')`` and everything after it stops,
    with nothing in the server log.  Catching it here costs milliseconds.
    """

    def setUp(self):
        from pdbenergy.gui import INDEX_HTML

        self.html = INDEX_HTML

    def test_every_id_the_script_uses_exists_in_the_markup(self):
        import re

        declared = set(re.findall(r'id="([^"]+)"', self.html))
        used = set(re.findall(r"\$\('([^']+)'\)", self.html))
        used |= set(re.findall(r"getElementById\('([^']+)'\)", self.html))
        missing = sorted(used - declared)
        self.assertEqual(missing, [], f"script references missing element ids: {missing}")

    def test_tab_panels_match_the_nav_buttons(self):
        """The nav builds section ids as 'tab-' + data-tab, so both sides must agree."""
        import re

        tabs = set(re.findall(r'data-tab="([^"]+)"', self.html))
        self.assertTrue(tabs, "no nav buttons found")
        panels = set(re.findall(r'id="tab-([^"]+)"', self.html))
        self.assertEqual(tabs, panels,
                         f"nav buttons {sorted(tabs)} vs panels {sorted(panels)}")

    def test_javascript_is_not_obviously_broken(self):
        """Brace/paren balance over the script block, ignoring strings and comments."""
        import re

        script = self.html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        stripped = re.sub(r"//[^\n]*", "", script)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
        stripped = re.sub(r"'(?:\\.|[^'\\])*'", "''", stripped)
        stripped = re.sub(r'"(?:\\.|[^"\\])*"', '""', stripped)
        for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
            self.assertEqual(
                stripped.count(opener), stripped.count(closer),
                f"unbalanced {opener}{closer} in the front-end script",
            )

    def test_server_binds_loopback_by_default(self):
        """There is no authentication, so the default host must stay on loopback."""
        from pdbenergy.gui import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.host, "127.0.0.1")

    def test_path_helper_refuses_to_escape_its_root(self):
        import tempfile

        from pdbenergy.gui import Handler

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "eval"), exist_ok=True)
            helper = Handler._safe_join
            # ``self`` is only used for nothing here, so pass None via the function.
            self.assertIsNotNone(helper(None, root, "eval/ok.png"))
            for bad in ("../x", "../../x", "eval/../../x", "a/../../../etc/passwd"):
                self.assertIsNone(helper(None, root, bad), f"{bad} escaped the root")


if __name__ == "__main__":
    unittest.main()
