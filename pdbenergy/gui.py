"""A local web GUI for the whole pipeline: data -> train -> evaluate -> predict.

Why a browser front end built on the standard library
----------------------------------------------------
* **Zero new dependencies.**  ``http.server`` and ``subprocess`` are stdlib, so
  the GUI works offline and adds nothing to ``requirements.txt`` - consistent
  with the rest of the project, which deliberately has no graph library and no
  plotting framework beyond Matplotlib.
* **Long jobs must not freeze the interface.**  Labelling a protein takes
  minutes; training takes tens of minutes.  Every job is therefore a *separate
  process* launched through the existing CLI, with its stdout streamed back
  line by line.  That gives live logs, a working Cancel button, and no GIL or
  blocking-IO problems.
* **One source of truth.**  The GUI never reimplements pipeline logic; it builds
  a command line and runs it.  Anything the CLI can do is reproducible from the
  log the GUI shows, which is exactly what you want when a run goes wrong.

Run it
------
    python -m pdbenergy.gui                 # serves http://127.0.0.1:8765
    python -m pdbenergy.gui --port 9000
    pdbenergy gui                           # same thing via the CLI

The server binds to 127.0.0.1 only.  There is no authentication because there is
no remote access; do not expose the port.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------- #
# Job manager
# --------------------------------------------------------------------------- #

#: How many output lines to keep per job.  Enough to scroll back through a
#: labelling run without letting a chatty job grow without bound.
LOG_RING = 4000


@dataclass
class Job:
    """One running (or finished) child process."""

    id: str
    label: str
    argv: list[str]
    cwd: str
    status: str = "running"            # running | done | failed | cancelled
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=LOG_RING))
    #: Total lines ever produced, so the client can ask for "everything after N"
    #: even when the ring buffer has already dropped the earliest ones.
    produced: int = 0
    process: subprocess.Popen | None = None

    def public(self, since: int = 0) -> dict[str, Any]:
        lines = list(self.lines)
        # Map the client's absolute cursor onto the ring buffer.
        dropped = max(0, self.produced - len(lines))
        start = max(0, since - dropped)
        return {
            "id": self.id,
            "label": self.label,
            "command": " ".join(self.argv),
            "status": self.status,
            "returncode": self.returncode,
            "started": self.started,
            "finished": self.finished,
            "elapsed": (self.finished or time.time()) - self.started,
            "lines": lines[start:],
            "cursor": self.produced,
            "dropped": dropped,
        }


class JobManager:
    """Launches CLI subprocesses and streams their output."""

    def __init__(self, cwd: str, python: str | None = None):
        self.cwd = cwd
        self.python = python or sys.executable
        #: The directory that contains both ``pdbenergy/`` and ``scripts/``.  It
        #: is put on the children's PYTHONPATH so jobs work even when the
        #: package is not installed, or when the editable install has gone stale
        #: (which happens silently if the checkout is moved or renamed - the
        #: generated path hook keeps pointing at the old location, and
        #: ``python -m pdbenergy.cli`` then fails with ModuleNotFoundError only
        #: from directories other than the project root).
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    # -- launching ---------------------------------------------------------- #
    def start(self, label: str, args: Sequence[str], module: str = "pdbenergy.cli") -> Job:
        """Run either ``python -m <module>`` or ``python <path.py>``.

        Most actions go through the CLI package, but a couple of useful tools
        live as standalone scripts (``scripts/measure_leakage.py``).  Detecting
        the ``.py`` suffix keeps one code path for both.
        """
        if module.endswith(".py"):
            argv = [self.python, "-u", module, *[str(a) for a in args]]
        else:
            argv = [self.python, "-u", "-m", module, *[str(a) for a in args]]
        job = Job(id=uuid.uuid4().hex[:12], label=label, argv=argv, cwd=self.cwd)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        existing = env.get("PYTHONPATH", "")
        parts = [self.project_root] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        # Merge stderr into stdout so the log reads in causal order.
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        job.process = subprocess.Popen(
            argv, cwd=self.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creationflags,
        )
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def _pump(self, job: Job) -> None:
        """Read the child's output on a thread until it exits."""
        assert job.process is not None
        try:
            assert job.process.stdout is not None
            for line in job.process.stdout:
                job.lines.append(line.rstrip("\n"))
                job.produced += 1
        except Exception as exc:                      # pragma: no cover
            job.lines.append(f"[gui] could not read output: {exc}")
            job.produced += 1
        finally:
            code = job.process.wait()
            job.returncode = code
            job.finished = time.time()
            if job.status != "cancelled":
                job.status = "done" if code == 0 else "failed"
            try:                                   # release the pipe promptly
                if job.process.stdout is not None:
                    job.process.stdout.close()
            except Exception:
                pass

    # -- inspection / control ----------------------------------------------- #
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.process is None or job.status != "running":
            return False
        job.status = "cancelled"
        job.process.terminate()          # SIGTERM / TerminateProcess
        try:
            job.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            job.process.kill()
        job.finished = time.time()
        job.lines.append("[gui] cancelled by user")
        job.produced += 1
        return True

    def running(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._order)
        return [self._jobs[i].public(0) for i in ids
                if self._jobs[i].status == "running"]

    def shutdown(self) -> int:
        """Terminate every still-running child.  Returns how many were stopped.

        Without this, closing the GUI leaves its children behind - and those
        children are OpenMM labelling runs that can occupy every core for hours.
        Terminating the parent does *not* cascade to the child processes.
        """
        stopped = 0
        for job in list(self._jobs.values()):
            if job.process is not None and job.status == "running":
                if self.cancel(job.id):
                    stopped += 1
        return stopped

    def recent(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
        out = []
        for i in ids:
            job = self._jobs[i]
            out.append({
                "id": job.id, "label": job.label, "status": job.status,
                "returncode": job.returncode,
                "elapsed": (job.finished or time.time()) - job.started,
            })
        return out


# --------------------------------------------------------------------------- #
# Project introspection (read-only, cheap)
# --------------------------------------------------------------------------- #


def _count(pattern_dir: str, suffix: str) -> int:
    if not os.path.isdir(pattern_dir):
        return 0
    return sum(1 for n in os.listdir(pattern_dir) if n.endswith(suffix))


def project_state(cfg, *, interim_dir: str, raw_dir: str, outputs_dir: str,
                  processed_dir: str) -> dict[str, Any]:
    """Everything the sidebar needs.  Deliberately cheap: counts and small JSONs.

    Parsing the raw PDB files is *not* done here - on a 2741-file directory that
    takes minutes.  The GUI exposes it as an explicit Scan job instead.
    """
    n_raw = _count(raw_dir, ".pdb") + _count(raw_dir, ".ent")
    n_labelled = _count(interim_dir, ".npz")

    inventory_path = os.path.join(outputs_dir, "inventory.json")
    inventory = None
    if os.path.exists(inventory_path):
        try:
            with open(inventory_path, "r", encoding="utf-8") as fh:
                reports = json.load(fh)
            usable = [r for r in reports if r.get("is_usable")]
            inventory = {
                "scanned": len(reports),
                "usable": len(usable),
                "unusable": len(reports) - len(usable),
                "min_residues": min((r["n_residues"] for r in usable), default=0),
                "max_residues": max((r["n_residues"] for r in usable), default=0),
                "scanned_at": os.path.getmtime(inventory_path),
            }
        except Exception:
            inventory = None

    return {
        "cwd": os.getcwd(),
        "counts": {
            "raw_pdb": n_raw,
            "labelled": n_labelled,
            "pending": max(0, n_raw - n_labelled),
        },
        "inventory": inventory,
        "runs": list_runs(outputs_dir),
        "defaults": {
            "prepare": {"max_residues": cfg.prepare.max_residues,
                        "min_residues": cfg.prepare.min_residues,
                        "ph": cfg.prepare.ph},
            "ensemble": {"n_torsion_per_level": cfg.ensemble.n_torsion_per_level,
                         "torsion_levels": len(cfg.ensemble.torsion_levels),
                         "md_temperatures": list(cfg.ensemble.md_temperatures),
                         "max_nmr_models": cfg.ensemble.max_nmr_models},
            "features": {"cutoff": cfg.features.cutoff, "n_rbf": cfg.features.n_rbf,
                         "max_neighbors": cfg.features.max_neighbors},
            "model": {"kind": cfg.model.kind, "hidden_dim": cfg.model.hidden_dim,
                      "n_interactions": cfg.model.n_interactions},
            "train": {"epochs": cfg.train.epochs, "batch_size": cfg.train.batch_size,
                      "learning_rate": cfg.train.learning_rate,
                      "patience": cfg.train.patience,
                      "val_fraction": cfg.train.val_fraction,
                      "test_fraction": cfg.train.test_fraction,
                      "split_mode": cfg.train.split_mode,
                      "target": cfg.train.target,
                      "loss": cfg.train.loss,
                      "cache_graphs": cfg.train.cache_graphs},
        },
        "figures": "outputs/<run>/eval/*.png",
    }


def list_runs(outputs_dir: str) -> list[dict[str, Any]]:
    """Every directory under outputs/ that looks like a trained run."""
    if not os.path.isdir(outputs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        checkpoint = os.path.join(run_dir, "checkpoint.pt")
        if not os.path.isdir(run_dir) or not os.path.exists(checkpoint):
            continue
        info: dict[str, Any] = {"name": name, "path": run_dir, "has_eval": False}
        for key, fname in (("summary", "data_summary.json"),
                           ("history", "history.json"),
                           ("config", "config.json")):
            path = os.path.join(run_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        info[key] = json.load(fh)
                except Exception:
                    pass
        metrics_path = os.path.join(run_dir, "eval", "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
                info["has_eval"] = True
                info["metrics"] = {
                    split: {
                        "mae": data.get("overall", {}).get("mae"),
                        "rmse": data.get("overall", {}).get("rmse"),
                        "r2": data.get("overall", {}).get("r2"),
                        "spearman": data.get("overall", {}).get("spearman"),
                        "rank_rho": (data.get("rank_discrimination", {}) or {}).get(
                            "mean_spearman"),
                    }
                    for split, data in (metrics.get("splits") or {}).items()
                }
                info["baselines"] = metrics.get("baselines") or {}
                info["n_parameters"] = metrics.get("n_parameters")
                info["figures"] = sorted(
                    f[:-4] for f in os.listdir(os.path.join(run_dir, "eval"))
                    if f.endswith(".png")
                )
            except Exception:
                pass
        if "summary" in info:
            info["n_proteins"] = info["summary"].get("n_proteins")
            info["n_samples"] = info["summary"].get("n_samples")
            info["splits"] = info["summary"].get("protein_splits")
        runs.append(info)
    return runs


def predict_checkpoints(outputs_dir: str) -> list[dict[str, Any]]:
    """Checkpoints usable for inference, best-by-validation first."""
    runs = [r for r in list_runs(outputs_dir) if os.path.exists(
        os.path.join(r["path"], "checkpoint.pt"))]
    runs.sort(key=lambda r: (
        (r.get("metrics", {}).get("val", {}) or {}).get("mae") or 1e9
    ))
    return [{"name": r["name"], "path": os.path.join(r["path"], "checkpoint.pt")}
            for r in runs]


def list_pdb_files(raw_dir: str, limit: int = 400) -> list[str]:
    if not os.path.isdir(raw_dir):
        return []
    names = sorted(n for n in os.listdir(raw_dir) if n.lower().endswith((".pdb", ".ent")))
    return names[:limit]


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

CONTENT_TYPES = {".png": "image/png", ".json": "application/json",
                 ".html": "text/html; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    server_version = "pdbenergy-gui"
    manager: JobManager
    cfg = None
    dirs: dict[str, str] = {}

    # -- helpers ------------------------------------------------------------ #
    def log_message(self, fmt, *args):        # keep the console quiet
        pass

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, body: str, status: int = 200, ctype: str = "text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _safe_join(self, root: str, relative: str) -> str | None:
        """Resolve ``relative`` under ``root``, refusing anything that escapes.

        The GUI serves files from outputs/ (figures, checkpoints) and data/raw.
        A crafted path must not be able to read arbitrary files, so the resolved
        path is checked against the root before use.
        """
        root_abs = os.path.abspath(root)
        candidate = os.path.abspath(os.path.join(root_abs, relative))
        if candidate == root_abs or candidate.startswith(root_abs + os.sep):
            return candidate
        return None

    # -- routing ------------------------------------------------------------ #
    def do_GET(self):                                          # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._text(INDEX_HTML)

        if path == "/api/state":
            return self._json(project_state(
                self.cfg,
                interim_dir=self.dirs["interim"],
                raw_dir=self.dirs["raw"],
                outputs_dir=self.dirs["outputs"],
                processed_dir=self.dirs["processed"],
            ))

        if path == "/api/jobs":
            return self._json({"running": self.manager.running(),
                               "recent": self.manager.recent()})

        if path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            job = self.manager.get(job_id)
            if job is None:
                return self._json({"error": "no such job"}, 404)
            since = int((query.get("since") or ["0"])[0])
            return self._json(job.public(since))

        if path == "/api/pdbs":
            return self._json({"files": list_pdb_files(self.dirs["raw"])})

        if path == "/api/checkpoints":
            return self._json({"checkpoints": predict_checkpoints(self.dirs["outputs"])})

        if path == "/api/inventory":
            reports = self._load_inventory()
            if reports is None:
                return self._json({"error": "no inventory yet - run a Scan"}, 404)
            max_res = int((query.get("max_residues") or ["120"])[0])
            min_res = int((query.get("min_residues") or ["10"])[0])
            labelled = self._labelled_ids()
            if query.get("pending") == ["1"]:
                # Only entries worth labelling that have no ensemble yet.
                rows = [
                    r for r in reports
                    if _usable_at(r, max_res, min_res) and r["pdb_id"] not in labelled
                ]
            else:
                rows = reports
            rows = sorted(rows, key=lambda r: r.get("n_residues", 0))
            return self._json({
                "n_rows": len(rows),
                "labelled": len(labelled),
                "rows": rows[: int((query.get("limit") or ["300"])[0])],
            })

        if path.startswith("/api/figure/"):
            # /api/figure/<run>/<name>
            parts = path[len("/api/figure/"):].split("/")
            if len(parts) != 2:
                return self._json({"error": "bad figure path"}, 400)
            run, name = parts
            target = self._safe_join(
                os.path.join(self.dirs["outputs"], run, "eval"), name + ".png"
            )
            if not target or not os.path.exists(target):
                return self._json({"error": "no such figure"}, 404)
            with open(target, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/predictions":
            target = os.path.join(self.dirs["outputs"], "predictions.json")
            if not os.path.exists(target):
                return self._json({"error": "no predictions yet"}, 404)
            try:
                with open(target, "r", encoding="utf-8") as fh:
                    return self._json({"rows": json.load(fh)})
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)

        if path == "/api/rounds":
            target = os.path.join(self.dirs["outputs"], "iterative", "rounds.jsonl")
            if not os.path.exists(target):
                return self._json({"rounds": []})
            rounds = []
            with open(target, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rounds.append(json.loads(line))
            return self._json({"rounds": rounds})

        return self._json({"error": "not found", "path": path}, 404)

    def do_POST(self):                                         # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._body()

        if path == "/api/job":
            return self._start_job(body)

        if path.startswith("/api/job/") and path.endswith("/cancel"):
            job_id = path[len("/api/job/"):-len("/cancel")]
            ok = self.manager.cancel(job_id)
            return self._json({"cancelled": ok})

        return self._json({"error": "not found", "path": path}, 404)

    # -- job construction --------------------------------------------------- #
    def _start_job(self, body: dict[str, Any]) -> None:
        """Turn a request from the UI into a CLI invocation.

        The GUI deliberately only chooses *arguments*; every decision about what
        the arguments mean lives in the CLI.  That keeps the log the GUI shows
        sufficient to reproduce the run by hand.
        """
        action = body.get("action")

        if action == "download":
            args = ["download"]
            ids = [i for i in (body.get("ids") or []) if i]
            if ids:
                args += ["--ids", *ids]
            if body.get("force"):
                args.append("--force")
            return self._ok("download", "pdbenergy.cli", args)

        if action == "scan":
            args = ["inventory", "--no-table",
                    "--max-residues", str(int(body.get("max_residues", 120))),
                    "--min-residues", str(int(body.get("min_residues", 10)))]
            return self._ok("scan raw entries", "pdbenergy.cli", args)

        if action == "label":
            ids = [i for i in (body.get("ids") or []) if i]
            if not ids:
                return self._json({"error": "no entries selected"}, 400)
            args = ["ensemble", "--ids", *ids,
                    "--threads", str(int(body.get("threads", 8))),
                    "--workers", str(int(body.get("workers", 1)))]
            if body.get("overwrite"):
                args.append("--overwrite")
            return self._ok(f"label {len(ids)} entries", "pdbenergy.cli", args)

        if action == "dataset":
            return self._ok("build dataset", "pdbenergy.cli", ["dataset"])

        if action == "train":
            args = ["train", "--model", str(body.get("model", "schnet"))]
            for flag, key, cast in (("--epochs", "epochs", int),
                                    ("--batch-size", "batch_size", int),
                                    ("--learning-rate", "learning_rate", float),
                                    ("--hidden-dim", "hidden_dim", int),
                                    ("--interactions", "interactions", int),
                                    ("--tag", "tag", str),
                                    ("--split-mode", "split_mode", str)):
                value = body.get(key)
                if value not in (None, "", 0, 0.0):
                    args += [flag, str(cast(value))]
            if body.get("init_from"):
                args += ["--init-from", str(body["init_from"])]
            if body.get("resume"):
                args += ["--resume", str(body["resume"])]
            if body.get("recompute_normalisation"):
                args.append("--recompute-normalisation")
            if body.get("no_cache_graphs"):
                args.append("--no-cache-graphs")
            tag = body.get("tag") or f"{body.get('model', 'schnet')}_protein"
            return self._ok(f"train {body.get('model', 'schnet')} -> {tag}",
                            "pdbenergy.cli", args)

        if action == "evaluate":
            run_dir = body.get("run_dir")
            if not run_dir:
                return self._json({"error": "no run selected"}, 400)
            return self._ok(f"evaluate {os.path.basename(run_dir)}", "pdbenergy.cli",
                            ["evaluate", "--run-dir", str(run_dir)])

        if action == "predict":
            paths = [p for p in (body.get("paths") or []) if p]
            if not paths:
                return self._json({"error": "no PDB files given"}, 400)
            checkpoint = body.get("checkpoint")
            if not checkpoint:
                return self._json({"error": "no checkpoint selected"}, 400)
            args = ["predict", *paths, "--checkpoint", str(checkpoint),
                    "--max-models", str(int(body.get("max_models", 1))),
                    "--threads", str(int(body.get("threads", 4))),
                    "--json", os.path.join(self.dirs["outputs"], "predictions.json"),
                    "--quiet"]
            if body.get("verify"):
                args.append("--verify")
            return self._ok(f"predict {len(paths)} file(s)", "pdbenergy.cli", args)

        if action == "ablate":
            args = ["ablate", "--epochs", str(int(body.get("epochs", 20))),
                    "--model", str(body.get("model", "mlp"))]
            return self._ok("split-leakage ablation", "pdbenergy.cli", args)

        if action == "measure_leakage":
            return self._ok("leakage measurement",
                            os.path.join("scripts", "measure_leakage.py"), [])
        if action == "estimate_workload":
            return self._ok("workload estimate",
                            os.path.join("scripts", "estimate_workload.py"), [])

        if action == "iterate":
            args = ["--rounds", str(int(body.get("rounds", 1)))]
            if body.get("build_new", True):
                args.append("--build-new")
            else:
                args.append("--no-build-new")
            if body.get("build_limit"):
                args += ["--build-limit", str(int(body["build_limit"]))]
            if body.get("max_residues"):
                args += ["--max-residues", str(int(body["max_residues"]))]
            if not body.get("warm_start", True):
                args.append("--no-warm-start")
            if body.get("epochs"):
                args += ["--epochs", str(int(body["epochs"]))]
            if body.get("model"):
                args += ["--model", str(body["model"])]
            args += ["--threads", str(int(body.get("threads", 8))),
                     "--workers", str(int(body.get("workers", 1)))]
            return self._ok(f"iterate x{body.get('rounds', 1)}", "pdbenergy.iterate", args)

        return self._json({"error": f"unknown action {action!r}"}, 400)

    def _ok(self, label: str, module: str, args: list[str]) -> None:
        job = self.manager.start(label, args, module=module)
        self._json({"job": job.public(0), "label": label})

    # -- small helpers ------------------------------------------------------ #
    def _load_inventory(self):
        path = os.path.join(self.dirs["outputs"], "inventory.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _labelled_ids(self) -> set[str]:
        interim = self.dirs["interim"]
        if not os.path.isdir(interim):
            return set()
        return {n[:-4] for n in os.listdir(interim) if n.endswith(".npz")}


def _usable_at(report: dict, max_residues: int, min_residues: int) -> bool:
    """Re-apply the entry filter for the *current* caps.

    ``inventory.json`` stores ``is_usable`` from whatever caps that scan used, so
    it must be recomputed to honour the sliders in the UI.  The ``reason`` field
    still says why an entry failed, which is what distinguishes "blocked by the
    cap" (recoverable) from "no amino acids" (never usable).
    """
    reason = report.get("reason", "") or ""
    if reason.startswith("parse error"):
        return False
    if "no standard amino-acid" in reason or "multi-chain" in reason:
        return False
    return min_residues <= report.get("n_residues", 0) <= max_residues


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pdbenergy — 训练 / 预测 / 测试</title>
<style>
:root{
  --bg:#0f1419; --panel:#161c24; --panel2:#1d242e; --line:#2a3441;
  --fg:#dde3ea; --dim:#8b98a8; --accent:#4da3ff; --ok:#3fb950; --warn:#d29922;
  --bad:#f85149; --mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;
  background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
header .spacer{flex:1}
.pill{font:12px/1 var(--mono);padding:5px 9px;border-radius:999px;
  background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
.pill b{color:var(--fg);font-weight:600}
main{display:grid;grid-template-columns:230px 1fr;min-height:calc(100vh - 48px)}
nav{padding:14px 10px;border-right:1px solid var(--line);background:var(--panel)}
nav button{display:block;width:100%;text-align:left;margin-bottom:4px;padding:9px 12px;
  background:transparent;border:1px solid transparent;border-radius:7px;color:var(--dim);
  font:14px system-ui;cursor:pointer}
nav button:hover{background:var(--panel2);color:var(--fg)}
nav button.on{background:var(--panel2);border-color:var(--accent);color:var(--fg)}
section{padding:18px 22px;display:none}
section.on{display:block}
h2{font-size:16px;margin:0 0 4px;font-weight:600}
h3{font-size:13px;margin:20px 0 8px;color:var(--dim);font-weight:600;
  text-transform:uppercase;letter-spacing:.6px}
p.hint{color:var(--dim);font-size:12.5px;margin:2px 0 14px;max-width:80ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:14px 16px;margin-bottom:14px}
.grid{display:grid;gap:10px 14px;grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:4px}
input,select{width:100%;padding:7px 9px;background:var(--bg);color:var(--fg);
  border:1px solid var(--line);border-radius:6px;font:13px var(--mono)}
input[type=checkbox]{width:auto}
button.act{padding:8px 15px;border-radius:7px;border:1px solid var(--accent);
  background:#153055;color:#cfe6ff;font:13px system-ui;cursor:pointer}
button.act:hover{background:#1b3d69}
button.act.ghost{border-color:var(--line);background:var(--panel2);color:var(--fg)}
button.act.danger{border-color:var(--bad);background:#3a1a1a;color:#ffd7d5}
button.act:disabled{opacity:.45;cursor:not-allowed}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font:12.5px var(--mono)}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--panel)}
tbody tr:hover{background:var(--panel2)}
.scroll{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:7px}
#log{font:12px/1.5 var(--mono);background:#0a0e12;border:1px solid var(--line);
  border-radius:7px;padding:10px;height:230px;overflow:auto;white-space:pre-wrap;
  word-break:break-word}
#logwrap{padding:0 22px 20px}
.badge{display:inline-block;padding:2px 7px;border-radius:5px;font-size:11px;
  font-family:var(--mono)}
.b-ok{background:#12351c;color:var(--ok)} .b-bad{background:#3a1a1a;color:var(--bad)}
.b-run{background:#153055;color:var(--accent)} .b-idle{background:var(--panel2);color:var(--dim)}
.figs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.figs figure{margin:0;background:var(--panel2);border:1px solid var(--line);
  border-radius:7px;padding:8px}
.figs img{width:100%;border-radius:5px;background:#fff}
.figs figcaption{font:11px var(--mono);color:var(--dim);margin-top:6px;text-align:center}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font:12.5px var(--mono)}
.kv div:nth-child(odd){color:var(--dim)}
.warn{color:var(--warn)} .bad{color:var(--bad)} .ok{color:var(--ok)}
</style>
</head>
<body>
<header>
  <h1>pdbenergy</h1>
  <span class="pill">raw <b id="s-raw">–</b></span>
  <span class="pill">labelled <b id="s-lab">–</b></span>
  <span class="pill">pending <b id="s-pend">–</b></span>
  <span class="pill">inventory <b id="s-inv">–</b></span>
  <span class="spacer"></span>
  <span class="pill" id="s-job">idle</span>
  <button class="act ghost" id="btn-cancel" disabled>取消</button>
  <button class="act ghost" onclick="refresh()">刷新</button>
</header>
<main>
<nav>
  <button data-tab="overview" class="on">概览</button>
  <button data-tab="data">数据 / 打标签</button>
  <button data-tab="train">训练</button>
  <button data-tab="eval">评估 / 测试</button>
  <button data-tab="predict">预测</button>
  <button data-tab="iterate">迭代训练</button>
</nav>
<div>
  <section id="tab-overview" class="on">
    <h2>概览</h2>
    <p class="hint">项目当前状态。所有面板共享底部同一个实时日志。</p>
    <div class="card"><div class="kv" id="ov-kv"></div></div>
    <h3>已有运行</h3>
    <div class="scroll"><table id="ov-runs"></table></div>
  </section>

  <section id="tab-data">
    <h2>数据 / 打标签</h2>
    <p class="hint">扫描会解析 data/raw 里每个文件（几千个文件要几分钟），过滤掉非蛋白、
      过短和过长的条目。扫描结果决定哪些条目值得花 OpenMM 时间。打标签是最慢的一步，
      实测每个蛋白 2–4 分钟；已完成的会跳过，可以分批做。</p>
    <div class="card">
      <div class="grid">
        <div><label>max_residues</label><input id="d-max" type="number" value="120"></div>
        <div><label>min_residues</label><input id="d-min" type="number" value="10"></div>
        <div><label>OpenMM threads</label><input id="d-threads" type="number" value="8"></div>
        <div><label>并行 workers</label><input id="d-workers" type="number" value="1"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" onclick="post('scan',{max_residues:+d('d-max'),min_residues:+d('d-min')})">扫描 data/raw</button>
        <button class="act" onclick="loadPending()">列出待标注</button>
        <button class="act ghost" onclick="post('download',{})">下载内置条目</button>
        <button class="act ghost" onclick="post('dataset',{})">构建数据集切分</button>
        <button class="act ghost" onclick="post('estimate_workload',{})">工作量估算</button>
      </div>
      <p class="hint" style="margin:12px 0 0">提示：workers × threads 不要超过物理核数
        （4 核用 <code>4 × 2</code>）。</p>
    </div>
    <h3>条目 <span id="d-count" class="badge b-idle">未加载</span></h3>
    <div class="row" style="margin-bottom:8px">
      <button class="act ghost" onclick="selectAll(true)">全选</button>
      <button class="act ghost" onclick="selectAll(false)">清空</button>
      <input id="d-limit" type="number" value="50" style="width:90px">
      <button class="act" onclick="labelling()">标注已勾选</button>
    </div>
    <div class="scroll"><table id="d-table"></table></div>
  </section>

  <section id="tab-train">
    <h2>训练</h2>
    <p class="hint">超参改动会写进命令行并显示在日志里，所以每次运行都可手工复现。
      改 <code>hidden/cutoff/n_rbf</code> 会使图缓存失效并改变权重形状——那种情况下不能热启动。</p>
    <div class="card">
      <div class="grid">
        <div><label>model</label><select id="t-model">
          <option value="schnet">schnet (图神经网络)</option>
          <option value="mlp">mlp (描述符基线)</option></select></div>
        <div><label>epochs（本次）</label><input id="t-epochs" type="number" value="25"></div>
        <div><label>batch_size</label><input id="t-batch" type="number" value="16"></div>
        <div><label>learning_rate</label><input id="t-lr" type="number" step="0.0001" value="0.0005"></div>
        <div><label>hidden_dim</label><input id="t-hidden" type="number" value="64"></div>
        <div><label>n_interactions</label><input id="t-int" type="number" value="3"></div>
        <div><label>split_mode</label><select id="t-split">
          <option value="protein">protein（正确）</option>
          <option value="frame">frame（泄漏，仅用于对照）</option></select></div>
        <div><label>tag（输出目录名）</label><input id="t-tag" value="schnet_protein"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="t-nocache"> 禁用图缓存（大数据集）</label>
        <label style="margin:0"><input type="checkbox" id="t-recompute"> 重算归一化</label>
      </div>
      <div class="grid" style="margin-top:12px">
        <div><label>热启动 --init-from（数据变多时用）</label><input id="t-init" placeholder="outputs/.../checkpoint.pt"></div>
        <div><label>续训 --resume（中断后接着跑）</label><input id="t-resume" placeholder="outputs/某run目录"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" onclick="train()">开始训练</button>
        <span class="hint" style="margin:0">两者只能填一个</span>
      </div>
    </div>
    <div id="t-recent"></div>
  </section>

  <section id="tab-eval">
    <h2>评估 / 测试</h2>
    <p class="hint">选一个运行做评估并看图。注意：<b>MAE 单独看会骗人</b>——
      务必同时看「预测跨度」和「组内 ρ」。预测跨度只有百分之几，说明模型基本在输出常数。</p>
    <div class="card">
      <div class="grid">
        <div><label>运行</label><select id="e-run"></select></div>
        <div><label>消融 epochs</label><input id="e-epochs" type="number" value="20"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" onclick="evaluate()">评估该运行</button>
        <button class="act ghost" onclick="post('ablate',{epochs:+d('e-epochs'),model:'mlp'})">数据泄漏消融</button>
      </div>
    </div>
    <div class="card" id="e-metrics"></div>
    <div class="figs" id="e-figs"></div>
  </section>

  <section id="tab-predict">
    <h2>预测</h2>
    <p class="hint">给 PDB 文件打分并排序。预测输出的是相对构象能 ΔE =
      E − min(E)，只在同一分子内部可比。<code>verify</code> 会额外算一次真值力场能量，
      可以直接看到误差（需要 OpenMM，慢一些）。</p>
    <div class="card">
      <div class="grid">
        <div><label>checkpoint</label><select id="p-ckpt"></select></div>
        <div><label>每个文件取前 N 个模型</label><input id="p-models" type="number" value="1"></div>
        <div><label>OpenMM threads（verify 用）</label><input id="p-threads" type="number" value="4"></div>
        <div><label>或手填路径（逗号分隔）</label><input id="p-paths" placeholder="data/raw/1L2Y.pdb"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="p-verify"> --verify 对照真值</label>
        <button class="act" onclick="predict()">预测</button>
        <button class="act ghost" onclick="loadPredictions()">读取上次结果</button>
      </div>
      <div class="grid" style="margin-top:12px">
        <div><label>从 data/raw 选择文件</label><select id="p-file" size="6" multiple></select></div>
      </div>
    </div>
    <div class="scroll"><table id="p-table"></table></div>
  </section>

  <section id="tab-iterate">
    <h2>迭代训练</h2>
    <p class="hint">每轮：标注新条目 → 重建数据集 → 从上一轮热启动 → 评估 → 记录。
      切分在第 1 轮冻结，之后新增的蛋白只进 train，所以跨轮指标可比。
      <code>build-limit</code> 用来分批，避免一轮就把几百条全标了。</p>
    <div class="card">
      <div class="grid">
        <div><label>rounds</label><input id="i-rounds" type="number" value="1"></div>
        <div><label>本轮最多标注 N 条（0=全部）</label><input id="i-limit" type="number" value="20"></div>
        <div><label>max_residues</label><input id="i-max" type="number" value="200"></div>
        <div><label>epochs / 轮</label><input id="i-epochs" type="number" value="20"></div>
        <div><label>model</label><select id="i-model">
          <option value="schnet">schnet</option><option value="mlp">mlp</option></select></div>
        <div><label>threads</label><input id="i-threads" type="number" value="8"></div>
        <div><label>workers</label><input id="i-workers" type="number" value="1"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="i-build" checked> 标注新条目</label>
        <label style="margin:0"><input type="checkbox" id="i-warm" checked> 热启动</label>
        <button class="act" onclick="iterate()">运行迭代</button>
        <button class="act ghost" onclick="loadRounds()">读取轮次表</button>
      </div>
    </div>
    <div id="i-table"></div>
  </section>
</div>
</main>
<div id="logwrap">
  <h3 style="margin-top:0">实时日志 <span id="log-label" class="badge b-idle">无任务</span></h3>
  <div id="log">（尚未运行任何任务）</div>
</div>

<script>
const $ = id => document.getElementById(id);
const d = id => $(id).value;
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num = (v,n=2) => (v==null||isNaN(v)) ? '—' : Number(v).toFixed(n);

let STATE = null, ACTIVE = null, CURSOR = 0, POLL = null;

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  $('tab-' + b.dataset.tab).classList.add('on');
  if (b.dataset.tab === 'eval') fillRuns();
  if (b.dataset.tab === 'predict') { fillCheckpoints(); fillFiles(); }
});

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let t = await r.text(); throw new Error(r.status + ' ' + t.slice(0,200)); }
  return r.json();
}

async function refresh() {
  try {
    STATE = await api('/api/state');
    const c = STATE.counts;
    $('s-raw').textContent = c.raw_pdb;
    $('s-lab').textContent = c.labelled;
    $('s-pend').textContent = c.pending;
    $('s-inv').textContent = STATE.inventory
      ? `${STATE.inventory.usable}/${STATE.inventory.scanned} usable` : '未扫描';
    renderOverview();
    fillRuns();
    fillCheckpoints();
  } catch (e) { log('refresh 失败: ' + e.message); }
}

function renderOverview() {
  if (!STATE) return;
  const c = STATE.counts, inv = STATE.inventory;
  const rows = [
    ['工作目录', STATE.cwd],
    ['data/raw 里的 PDB', c.raw_pdb],
    ['已打标签 (data/interim)', c.labelled],
    ['待标注', c.pending],
    ['体检', inv ? `${inv.scanned} 条，可用 ${inv.usable}，排除 ${inv.unusable}` : '未扫描'],
    ['残基范围（可用）', inv ? `${inv.min_residues} – ${inv.max_residues}` : '—'],
  ];
  $('ov-kv').innerHTML = rows.map(([k,v]) =>
    `<div>${esc(k)}</div><div>${esc(v)}</div>`).join('');

  const runs = STATE.runs || [];
  let h = `<thead><tr><th>运行</th><th>参数</th><th>蛋白</th><th>样本</th>
    <th>val MAE</th><th>test MAE</th><th>test ρ</th><th>预测跨度</th></tr></thead><tbody>`;
  if (!runs.length) h += `<tr><td colspan="8" style="color:var(--dim)">还没有训练好的运行</td></tr>`;
  for (const r of runs) {
    const m = (r.metrics || {}), v = m.val || {}, t = m.test || {};
    const span = r.has_eval ? '' : '';
    h += `<tr><td>${esc(r.name)}</td><td>${esc((r.summary&&r.summary.split_mode)||'')}
      /${esc((r.summary&&r.summary.target)||'')}</td>
      <td>${r.n_proteins ?? '—'}</td><td>${r.n_samples ?? '—'}</td>
      <td>${num(v.mae)}</td><td>${num(t.mae)}</td>
      <td>${num(t.spearman,3)}</td>
      <td>${r.has_eval ? '<span class="ok">已评估</span>' : '<span class="dim">未评估</span>'}</td>
      </tr>`;
  }
  $('ov-runs').innerHTML = h + '</tbody>';
}

function fillRuns() {
  if (!STATE) return;
  const sel = $('e-run'); const cur = sel.value;
  sel.innerHTML = (STATE.runs||[]).map(r =>
    `<option value="${esc(r.path)}">${esc(r.name)}${r.has_eval?' ✓':''}</option>`).join('');
  if (cur) sel.value = cur;
  renderMetrics();
  renderRecent();
}

function renderMetrics() {
  if (!STATE) return;
  const name = ($('e-run').value||'').split(/[\\/]/).pop();
  const r = (STATE.runs||[]).find(x => x.name === name);
  if (!r) { $('e-metrics').innerHTML = '<span class="hint">未选择运行</span>'; $('e-figs').innerHTML=''; return; }
  if (!r.has_eval) { $('e-metrics').innerHTML = '<span class="hint">该运行还没评估过，点上面的「评估该运行」。</span>'; $('e-figs').innerHTML=''; return; }
  const m = r.metrics || {}, base = r.baselines || {};
  let h = `<div class="kv"><div>参数量</div><div>${r.n_parameters ?? '—'}</div>`;
  for (const s of ['val','test']) {
    const x = m[s]; if (!x) continue;
    const b = (base[s]||{}).mae;
    const verdict = (b != null && x.mae != null)
      ? (x.mae < b ? '<span class="ok">优于常数基线</span>' : '<span class="bad">不如常数基线</span>') : '';
    h += `<div>${s} MAE / RMSE</div><div>${num(x.mae)} / ${num(x.rmse)} ${verdict}</div>`;
    h += `<div>${s} R² / Spearman</div><div>${num(x.r2,4)} / ${num(x.spearman,4)}</div>`;
    h += `<div>${s} 组内平均 ρ</div><div>${num(x.rank_rho,4)} ${x.rank_rho!=null && x.rank_rho<0.3 ? '<span class="warn">← 排序能力弱</span>':''}</div>`;
    if (b != null) h += `<div>${s} 常数基线 MAE</div><div>${num(b)}</div>`;
  }
  h += '</div><p class="hint" style="margin:12px 0 0">预测跨度不在 metrics.json 里，'+
       '看 parity 图或跑 <code>scripts/compare_configs.py</code> 的输出来确认模型是否在输出常数。</p>';
  $('e-metrics').innerHTML = h;
  $('e-figs').innerHTML = (r.figures||[]).map(f =>
    `<figure><img src="/api/figure/${encodeURIComponent(r.name)}/${encodeURIComponent(f)}" loading="lazy">
     <figcaption>${esc(f)}</figcaption></figure>`).join('');
}

function renderRecent() {
  const wrap = $('t-recent');
  if (!STATE) return;
  const runs = STATE.runs||[];
  if (!runs.length) { wrap.innerHTML=''; return; }
  let h = '<h3>最近运行的学习曲线</h3><div class="figs">';
  for (const r of runs.slice(-3)) {
    if ((r.figures||[]).includes('learning_curve'))
      h += `<figure><img src="/api/figure/${encodeURIComponent(r.name)}/learning_curve">
        <figcaption>${esc(r.name)}</figcaption></figure>`;
  }
  wrap.innerHTML = h + '</div>';
}

/* ---- data tab ---- */
async function loadPending() {
  const q = `?pending=1&max_residues=${d('d-max')}&min_residues=${d('d-min')}&limit=300`;
  try {
    const r = await api('/api/inventory' + q);
    renderEntries(r.rows, `待标注 ${r.n_rows}`);
  } catch (e) {
    log('读取 inventory 失败：' + e.message + ' —— 先点「扫描 data/raw」');
  }
}
function renderEntries(rows, label) {
  $('d-count').textContent = label;
  $('d-count').className = 'badge b-run';
  let h = `<thead><tr><th>选</th><th>PDB</th><th>残基</th><th>原子</th><th>模型</th>
    <th>实验</th><th>原因</th></tr></thead><tbody>`;
  rows.forEach((r,i) => {
    h += `<tr><td><input type="checkbox" class="pick" data-id="${esc(r.pdb_id)}"></td>
      <td>${esc(r.pdb_id)}</td><td>${r.n_residues}</td><td>${r.n_atoms}</td>
      <td>${r.n_models}</td><td>${esc((r.experiment||'').slice(0,22))}</td>
      <td>${esc(r.reason||'')}</td></tr>`;
  });
  $('d-table').innerHTML = h + '</tbody>';
}
function selectAll(on) { document.querySelectorAll('.pick').forEach(c => c.checked = on); }
function labelling() {
  const ids = [...document.querySelectorAll('.pick:checked')].map(c => c.dataset.id);
  if (!ids.length) return log('没有勾选任何条目');
  const limit = +d('d-limit');
  const chosen = limit > 0 ? ids.slice(0, limit) : ids;
  post('label', {ids: chosen, threads: +d('d-threads'), workers: +d('d-workers')});
}

/* ---- train tab ---- */
function train() {
  const init = d('t-init').trim(), resume = d('t-resume').trim();
  if (init && resume) return log('热启动和续训只能填一个');
  post('train', {
    model: d('t-model'), epochs: +d('t-epochs'), batch_size: +d('t-batch'),
    learning_rate: +d('t-lr'), hidden_dim: +d('t-hidden'),
    interactions: +d('t-int'), split_mode: d('t-split'), tag: d('t-tag').trim(),
    init_from: init || null, resume: resume || null,
    no_cache_graphs: $('t-nocache').checked,
    recompute_normalisation: $('t-recompute').checked,
  });
}
function evaluate() {
  const run = $('e-run').value;
  if (!run) return log('没有选择运行');
  post('evaluate', {run_dir: run});
}
function predict() {
  let paths = [...$('p-file').selectedOptions].map(o => o.value);
  const manual = d('p-paths').trim();
  if (manual) paths = paths.concat(manual.split(',').map(s => s.trim()).filter(Boolean));
  if (!paths.length) return log('没有选择 PDB 文件');
  const ck = $('p-ckpt').value;
  if (!ck) return log('没有可用 checkpoint');
  post('predict', {paths, checkpoint: ck, max_models: +d('p-models'),
                   threads: +d('p-threads'), verify: $('p-verify').checked});
}
async function fillCheckpoints() {
  try {
    const r = await api('/api/checkpoints');
    const sel = $('p-ckpt'), cur = sel.value;
    sel.innerHTML = r.checkpoints.map(c =>
      `<option value="${esc(c.path)}">${esc(c.name)}</option>`).join('') ||
      '<option value="">（无 checkpoint，先训练）</option>';
    if (cur) sel.value = cur;
  } catch(e) {}
}
async function fillFiles() {
  try {
    const r = await api('/api/pdbs');
    $('p-file').innerHTML = r.files.map(f =>
      `<option value="data/raw/${esc(f)}">${esc(f)}</option>`).join('');
  } catch(e) {}
}
async function loadPredictions() {
  try {
    const r = await api('/api/predictions');
    let h = `<thead><tr><th>文件</th><th>模型</th><th>预测 ΔE</th><th>真值 ΔE</th>
      <th>误差</th><th>残基</th></tr></thead><tbody>`;
    for (const p of r.rows) {
      h += `<tr><td>${esc(p.name)}</td><td>${p.model_index}</td>
        <td>${num(p.predicted_relative_energy_kcal_per_mol,3)}</td>
        <td>${p.true_relative_energy_kcal_per_mol==null?'—':num(p.true_relative_energy_kcal_per_mol,3)}</td>
        <td>${p.error_kcal_per_mol==null?'—':num(p.error_kcal_per_mol,3)}</td>
        <td>${p.n_residues}</td></tr>`;
    }
    $('p-table').innerHTML = h + '</tbody>';
  } catch (e) { log('没有可读的预测结果'); }
}
function iterate() {
  post('iterate', {
    rounds: +d('i-rounds'), build_limit: +d('i-limit'), max_residues: +d('i-max'),
    epochs: +d('i-epochs'), model: d('i-model'),
    threads: +d('i-threads'), workers: +d('i-workers'),
    build_new: $('i-build').checked, warm_start: $('i-warm').checked,
  });
}
async function loadRounds() {
  try {
    const r = await api('/api/rounds');
    if (!r.rounds.length) { $('i-table').innerHTML = '<p class="hint">还没有轮次记录</p>'; return; }
    let h = `<div class="scroll"><table><thead><tr><th>轮</th><th>蛋白</th>
      <th>train/val/test</th><th>val MAE</th><th>test MAE</th><th>test ρ</th>
      <th>组内 ρ</th><th>常数基线</th><th>跨度</th></tr></thead><tbody>`;
    for (const x of r.rounds) {
      h += `<tr><td>${x.round}</td><td>${x.n_proteins}</td>
        <td>${x.n_train}/${x.n_val}/${x.n_test}</td>
        <td>${num(x.val_mae)}</td><td>${num(x.test_mae)}</td>
        <td>${num(x.test_spearman,3)}</td><td>${num(x.within_protein_rho,3)}</td>
        <td>${num(x.constant_baseline_mae)}</td>
        <td>${x.prediction_span_ratio==null?'—':(100*x.prediction_span_ratio).toFixed(0)+'%'}</td>
        </tr>`;
    }
    $('i-table').innerHTML = h + '</tbody></table></div>';
  } catch(e) { log('读取轮次失败: ' + e.message); }
}

/* ---- jobs & log ---- */
async function post(action, payload) {
  try {
    const r = await api('/api/job', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(Object.assign({action}, payload||{}))});
    startWatch(r.job);
  } catch (e) { log('启动任务失败: ' + e.message); }
}
function startWatch(job) {
  ACTIVE = job.id; CURSOR = 0;
  $('log').textContent = '';
  $('log-label').textContent = job.label;
  $('log-label').className = 'badge b-run';
  $('s-job').textContent = '运行中';
  $('btn-cancel').disabled = false;
  if (POLL) clearInterval(POLL);
  POLL = setInterval(poll, 700);
  poll();
}
async function poll() {
  if (!ACTIVE) return;
  try {
    const j = await api(`/api/job/${ACTIVE}?since=${CURSOR}`);
    if (j.dropped) $('log').textContent += `\n[gui] ${j.dropped} 行较早输出已被丢弃\n`;
    if (j.lines.length) {
      const el = $('log');
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent += j.lines.join('\n') + '\n';
      if (atBottom) el.scrollTop = el.scrollHeight;
    }
    CURSOR = j.cursor;
    $('s-job').textContent = `${j.status} ${j.elapsed.toFixed(0)}s`;
    if (j.status !== 'running') {
      clearInterval(POLL); POLL = null; ACTIVE = null;
      $('btn-cancel').disabled = true;
      $('log-label').textContent = j.status === 'done' ? '完成' : j.status;
      $('log-label').className = 'badge ' + (j.status==='done'?'b-ok':'b-bad');
      refresh();
      if (j.status === 'done') { loadPredictions(); loadRounds(); }
    }
  } catch (e) { /* transient */ }
}
$('btn-cancel').onclick = async () => {
  if (!ACTIVE) return;
  await api(`/api/job/${ACTIVE}/cancel`, {method:'POST'});
};
function log(msg) { $('log').textContent += '\n' + msg; $('log').scrollTop = $('log').scrollHeight; }

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pdbenergy.gui",
        description="Local web GUI for the pdbenergy pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; keep it on loopback, there is no auth")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=None, help="JSON config for defaults")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not try to open a browser window")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config import Config

    cfg = Config.load(args.config) if args.config else Config()

    handler = type("BoundHandler", (Handler,), {
        "manager": JobManager(os.getcwd()),
        "cfg": cfg,
        "dirs": {
            "raw": args.raw_dir,
            "interim": args.interim_dir,
            "processed": args.processed_dir,
            "outputs": args.outputs_dir,
        },
    })

    httpd = None
    for port in range(args.port, args.port + 20):
        try:
            httpd = ThreadingHTTPServer((args.host, port), handler)
            break
        except OSError:
            continue
    if httpd is None:
        print(f"no free port in {args.port}..{args.port + 20}")
        return 1

    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"pdbenergy GUI  ->  {url}")
    print(f"  cwd            {os.getcwd()}")
    print(f"  data/raw       {args.raw_dir}  ({_count(args.raw_dir, '.pdb')} pdb files)")
    print(f"  data/interim   {args.interim_dir}  ({_count(args.interim_dir, '.npz')} labelled)")
    print(f"  outputs        {args.outputs_dir}")
    print("  Ctrl-C to stop.  Long jobs run as child processes and can be cancelled "
          "from the UI.")

    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stopped = handler.manager.shutdown()
        if stopped:
            print(f"terminated {stopped} running job(s)")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
