"""Both ways of starting a module must work.

    python -m pdbenergy.gui_qt        # the documented way
    python pdbenergy/gui_qt.py        # IDE "Run", double-click, absolute path

The second form has no parent package, so every module-level relative import
(``from .actions import ...``) fails with::

    ImportError: attempted relative import with no known parent package

That is exactly what a user hit.  Each entry-point module therefore re-dispatches
itself through the package when ``__package__`` is empty; these tests run the real
files as subprocesses to prove it, because the failure only appears in a fresh
interpreter - an in-process import would succeed either way.

``--help`` is used as the probe: argparse prints usage and exits before the
window is created or any work starts, so nothing blocks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: Modules with a __main__ guard that a user might reasonably launch directly.
ENTRY_MODULES = ("cli", "iterate", "gui", "gui_qt")


def run(args, *, cwd=ROOT):
    """Run a child process and decode its output as UTF-8.

    ``text=True`` alone decodes with the *system locale* codec, which is GBK on a
    Chinese Windows install - and the children print UTF-8.  That mismatch raises
    UnicodeDecodeError inside subprocess, leaving ``proc.stdout`` as None and
    making every assertion fail for a reason unrelated to what is being tested.
    Fixing the encoding on both sides is the only reliable way here.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=cwd, timeout=180,
                          env=env)


class TestDirectExecution(unittest.TestCase):
    def _assert_ok(self, proc, what):
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertNotIn("relative import", combined,
                         f"{what}: 相对导入失败\n{combined[-600:]}")
        self.assertNotIn("ImportError", combined,
                         f"{what}: 导入失败\n{combined[-600:]}")
        self.assertEqual(proc.returncode, 0,
                         f"{what}: 退出码 {proc.returncode}\n{combined[-600:]}")
        self.assertIn("usage", (proc.stdout or "").lower(), f"{what}: 没有打印用法")

    def test_modules_run_as_plain_scripts(self):
        """`python pdbenergy/<mod>.py --help` must work (the reported failure)."""
        for mod in ENTRY_MODULES:
            with self.subTest(module=mod):
                self._assert_ok(
                    run([os.path.join("pdbenergy", f"{mod}.py"), "--help"]),
                    f"script {mod}",
                )

    def test_modules_still_run_as_package_modules(self):
        """The documented `-m` form must not regress."""
        for mod in ENTRY_MODULES:
            with self.subTest(module=mod):
                self._assert_ok(run(["-m", f"pdbenergy.{mod}", "--help"]),
                                f"-m pdbenergy.{mod}")

    def test_scripts_work_from_an_unrelated_working_directory(self):
        """The bootstrap adds the *project* root, not the current directory.

        Running the file by absolute path from somewhere else must work too - the
        user may well have a shell open elsewhere.
        """
        with tempfile.TemporaryDirectory() as elsewhere:
            for mod in ENTRY_MODULES:
                with self.subTest(module=mod):
                    self._assert_ok(
                        run([os.path.join(ROOT, "pdbenergy", f"{mod}.py"), "--help"],
                            cwd=elsewhere),
                        f"script {mod} from {elsewhere}",
                    )

    def test_a_failing_subcommand_still_reports_normally(self):
        """Direct execution must not swallow the CLI's own error handling."""
        proc = run([os.path.join("pdbenergy", "cli.py"), "not-a-subcommand"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", (proc.stderr or "") + (proc.stdout or ""))

    def test_bootstrap_keeps_argv_intact(self):
        """Re-dispatching must not lose the command-line arguments."""
        proc = run([os.path.join("pdbenergy", "cli.py"), "inventory", "--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
        self.assertIn("--max-residues", proc.stdout)


class TestBootstrapIsPlacedCorrectly(unittest.TestCase):
    """The re-dispatch has to precede the first relative import, or it is useless."""

    def test_bootstrap_precedes_relative_imports(self):
        """The re-dispatch has to precede the first relative import, or it is useless.

        The search is deliberately not restricted to module level: a relative
        import *inside a function* fails just the same when the file is run as a
        script (gui.py keeps them all inside functions), so the bootstrap has to
        come first either way.
        """
        for mod in ENTRY_MODULES:
            with self.subTest(module=mod):
                with open(os.path.join(ROOT, "pdbenergy", f"{mod}.py"),
                          "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                bootstrap = next(
                    (i for i, line in enumerate(lines) if "__package__ in (None" in line),
                    None,
                )
                self.assertIsNotNone(bootstrap, f"{mod}.py 缺少引导块")
                first_relative = next(
                    (i for i, line in enumerate(lines)
                     if line.strip().startswith(("from .", "import ."))),
                    None,
                )
                self.assertIsNotNone(
                    first_relative,
                    f"{mod}.py 一处相对导入都没有 —— 那么这个模块本来就不需要引导块，"
                    f"应当从 ENTRY_MODULES 里去掉",
                )
                self.assertLess(bootstrap, first_relative,
                                f"{mod}.py 的引导块在第 {bootstrap+1} 行，"
                                f"晚于第 {first_relative+1} 行的相对导入，不起作用")

    def test_bootstrap_keeps_future_import_first(self):
        """`from __future__` must stay the first statement in the file."""
        for mod in ENTRY_MODULES:
            with self.subTest(module=mod):
                with open(os.path.join(ROOT, "pdbenergy", f"{mod}.py"),
                          "r", encoding="utf-8") as fh:
                    body = []
                    in_docstring = False
                    for raw in fh:
                        stripped = raw.strip()
                        if stripped.startswith('"""') and not in_docstring:
                            in_docstring = not stripped.endswith('"""') or stripped == '"""'
                            continue
                        if in_docstring:
                            if stripped.endswith('"""'):
                                in_docstring = False
                            continue
                        if stripped and not stripped.startswith("#"):
                            body.append(stripped)
                self.assertEqual(body[0], "from __future__ import annotations",
                                 f"{mod}.py 的第一条语句不是 __future__ 导入")


if __name__ == "__main__":
    unittest.main()
