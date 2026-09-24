"""Answer the question "is my GUI actually able to run anything?".

Motivation
----------
A GUI job once ended with ``✓ 任务完成`` in the log and nothing on disk, because
the child process could not import ``pdbenergy.cli`` (the checkout had been
renamed while the window was open) and still exited 0.  This script drives the
same :class:`pdbenergy.gui.JobManager` the GUIs use, so what it prints here is
what the interface sees.

Run from the project root::

    .venv\\Scripts\\python.exe scripts\\diagnose_launch.py

It is read-only apart from a scratch file under the system temp directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbenergy.gui import JobManager, project_root, resolve_dir  # noqa: E402

MISSING = "No module named pdbenergy.cli"


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def wait(manager: JobManager, job_id: str, timeout: float = 300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status != "running":
            return job
        time.sleep(0.05)
    raise SystemExit(f"job {job_id} did not finish within {timeout:.0f}s")


def main() -> int:
    manager = JobManager()          # no argument: falls back to the project root

    rule("1. Where the GUI thinks it is")
    print(f"  interpreter    : {manager.python}")
    print(f"  python version : {sys.version.split()[0]}")
    print(f"  process cwd    : {os.getcwd()}")
    print(f"  project root   : {project_root()}  (exists: {os.path.isdir(project_root())})")
    print(f"  job cwd        : {manager.cwd}  (exists: {os.path.isdir(manager.cwd)})")
    print(f"  PYTHONPATH     : {os.environ.get('PYTHONPATH', '<unset>')}")
    for key, path in (("raw", "data/raw"), ("interim", "data/interim"),
                      ("outputs", "outputs")):
        resolved = resolve_dir(path)
        print(f"  {key:<14}: {resolved}  (exists: {os.path.isdir(resolved)})")

    rule("2. Child-process import check (exactly what every job does first)")
    reason = manager.check_import()
    if reason is None:
        print("  OK - a child can import pdbenergy.cli.")
    else:
        print(f"  BROKEN - {reason}")
        print()
        for line in manager.import_hint("pdbenergy.cli", reason).splitlines():
            print(f"  {line}")
        return 1

    rule("3. A real job through the real manager")
    job = manager.start("diagnose --help", ["--help"])
    done = wait(manager, job.id)
    print(f"  command     : {done.public(0)['command']}")
    print(f"  status      : {done.status}")
    print(f"  return code : {done.returncode}")
    print("  first lines :")
    for line in list(done.lines)[:6]:
        print(f"      | {line}")

    rule("4. Regression guard: a child that exits 0 without importing")
    with tempfile.TemporaryDirectory() as scratch:
        liar = os.path.join(scratch, "liar.py")
        with open(liar, "w", encoding="utf-8") as fh:
            fh.write(f"print({MISSING!r})\n")
        bad = manager.start("simulated broken environment", [], module=liar)
        finished = wait(manager, bad.id, timeout=60)
        print(f"  child printed : {MISSING}")
        print(f"  return code   : {finished.returncode}   <- a clean exit code")
        print(f"  status        : {finished.status}   <- must be 'failed', not 'done'")
        print("  the log explains itself:")
        for line in list(finished.lines)[1:]:
            print(f"      | {line}")
        ok = finished.status == "failed"
    manager.shutdown()

    rule("Verdict")
    print("  The GUI can launch jobs." if ok else "  The GUI would report a false success!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
