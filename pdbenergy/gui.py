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

# --- 允许直接运行本文件：IDE 的 Run 按钮 / 双击 / python pdbenergy\gui.py ----- #
# 相对导入需要一个父包，直接跑文件没有，会报
#     ImportError: attempted relative import with no known parent package
# 所以先把自己当作包模块重新派发一次。必须在任何相对导入之前。
if __package__ in (None, ""):                                    # pragma: no cover
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pdbenergy.gui import main as _main

    _sys.exit(_main())

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
    #: The module (or script path) the child was asked to run; kept so a failed
    #: launch can name what could not be imported.
    module: str = ""
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


def project_root() -> str:
    """The directory that contains both ``pdbenergy/`` and ``scripts/``.

    Derived from this file's location, never from ``os.getcwd()``.  The GUI is
    routinely started from a desktop shortcut, an IDE or another shell, and a
    cwd that is not the project root silently sends every relative path
    (``data/raw``, ``data/interim``, ``outputs``) somewhere else - jobs then run
    "successfully" against an empty directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_dir(path: str, root: str | None = None) -> str:
    """Make a directory argument absolute, resisting a foreign launch cwd.

    The defaults are relative (``data/raw``) and must mean "next to the
    project", not "next to whatever directory the shortcut happened to start
    us in".
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root or project_root(), path))


#: What a Python child prints when it cannot find the module it was told to run.
#: ``python -m pkg.mod`` reports the *dotted* name even when the missing part is
#: the parent package, which is why a broken environment surfaces as a bare
#: ``No module named pdbenergy.cli`` with nothing to act on.
IMPORT_FAILURE_MARKERS = ("No module named ", "ModuleNotFoundError")


def running_under_debugger() -> str | None:
    """Name the debugger that has taken over this process, if there is one.

    This matters more than it looks.  A debugger does not just watch *this*
    process: VSCode's debugpy patches ``subprocess`` / ``os.exec*`` so that every
    child the GUI spawns is itself started under ``pydevd``.  Those children
    resolve ``-m`` through pydevd's bundled ``runpy``, whose
    ``_get_module_details`` reports a module it cannot find as
    ``ImportError("No module named X")`` - deliberately *without* a traceback -
    and the wrapper then exits 0.

    The result is indistinguishable from a broken install: every job fails with
    a bare ``No module named pdbenergy.cli`` and exit code 0, while the very same
    interpreter imports the package fine from ``-c`` or from a script.  The fix
    is not in the project - launch the GUI without the debugger.
    """
    if "pydevd" in sys.modules:
        return "pydevd（VSCode / PyCharm 调试器）"
    if "debugpy" in sys.modules:
        return "debugpy"
    # A debugger can also be present as the subprocess patch alone, without its
    # main module imported into this process.
    if any(name.endswith(("pydev_monkey", "pydevd_runpy", "pydevd"))
           for name in sys.modules):
        return "python 调试器（pydevd 的子进程补丁已生效）"
    if any(k.startswith(("PYDEVD_", "DEBUGPY_")) for k in os.environ):
        return "python 调试器（环境里带着 PYDEVD_* / DEBUGPY_*）"
    return None


class JobManager:
    """Launches CLI subprocesses and streams their output."""

    def __init__(self, cwd: str | None = None, python: str | None = None):
        #: The directory that contains both ``pdbenergy/`` and ``scripts/``.  It
        #: is put on the children's PYTHONPATH so jobs work even when the
        #: package is not installed, or when the editable install has gone stale
        #: (which happens silently if the checkout is moved or renamed - the
        #: generated path hook keeps pointing at the old location, and
        #: ``python -m pdbenergy.cli`` then fails with ModuleNotFoundError only
        #: from directories other than the project root).
        self.project_root = project_root()
        #: A cwd that does not exist makes ``Popen`` fail outright; a cwd that
        #: exists but is not the project root makes every relative data path
        #: wrong.  Falling back to the project root avoids both.
        self.cwd = cwd if cwd and os.path.isdir(cwd) else self.project_root
        self.python = python or sys.executable
        #: Which PYTHON* variables were stripped from the children, filled in by
        #: :meth:`child_env` so the log can say what was neutralised.
        self.dropped_env_keys: list[str] = []
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    #: Variables that change how a child interpreter starts up.  Every one of
    #: them is removed from the child's environment: an inherited value makes
    #: ``.venv\\Scripts\\python.exe`` behave like a different installation, and
    #: the failure looks like a missing package even though every path the GUI
    #: prints is correct.  ``PYTHONPATH`` is not in this list - we set it, and we
    #: keep any inherited entries behind the project root.
    POISONED_ENV_KEYS = ("PYTHONHOME", "PYTHONSAFEPATH", "PYTHONSTARTUP",
                         "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__",
                         "PYTHONNOUSERSITE", "PYTHONUSERBASE", "PYTHONPLATLIBDIR")

    #: Everything worth printing when a job fails.
    PYTHON_ENV_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH",
                       "PYTHONSTARTUP", "PYTHONNOUSERSITE", "PYTHONEXECUTABLE",
                       "__PYVENV_LAUNCHER__", "VIRTUAL_ENV", "PYTHONWARNINGS")

    def child_env(self) -> dict[str, str]:
        """The environment every child gets: project root first, no poison."""
        self.dropped_env_keys = sorted(k for k in self.POISONED_ENV_KEYS
                                       if os.environ.get(k))
        env = dict(os.environ)
        for key in self.dropped_env_keys:
            env.pop(key, None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        existing = env.get("PYTHONPATH", "")
        parts = [self.project_root] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def python_env_report(self) -> list[str]:
        """The Python-affecting variables, as the GUI process sees them."""
        lines = [f"[诊断]   {key}={os.environ[key]}"
                 for key in self.PYTHON_ENV_KEYS if os.environ.get(key)]
        if getattr(self, "dropped_env_keys", None):
            lines.append("[诊断]   已从子进程环境中移除："
                         + ", ".join(self.dropped_env_keys))
        return lines or ["[诊断]   （没有设置任何 PYTHON* 变量）"]

    def diagnose_child(self, module: str) -> list[str]:
        """Ask a real child what *it* sees.  Only runs after a failure.

        Reasoning from the outside had already proved useless: the interpreter
        path, the project root and the working directory were all correct, yet
        the import still failed.  This prints the child's own ``sys.path``, what
        it thinks about the package, and the real traceback - so the next
        failure names its own cause instead of repeating ``No module named``.
        """
        probe = (
            "import importlib.util as u, os, sys, traceback\n"
            "print('executable :', sys.executable)\n"
            "print('version    :', sys.version.split()[0])\n"
            "print('cwd        :', os.getcwd())\n"
            "print('prefix     :', sys.prefix)\n"
            "print('base_prefix:', sys.base_prefix)\n"
            "print('safe_path  :', sys.flags.safe_path)\n"
            "print('sys.path   :')\n"
            "for p in sys.path:\n"
            "    print('   ', repr(p), '->', 'dir' if p and os.path.isdir(p) else repr(p))\n"
            "for name in ('pdbenergy', 'pdbenergy.cli'):\n"
            "    try:\n"
            "        print('find_spec  :', name, '->', u.find_spec(name))\n"
            "    except Exception as exc:\n"
            "        print('find_spec  :', name, 'raised', type(exc).__name__, exc)\n"
            "print('cli.py     :', os.path.isfile(os.path.join(os.getcwd(),"
            " 'pdbenergy', 'cli.py')))\n"
            "try:\n"
            "    import pdbenergy.cli\n"
            "    print('import     : ok ->', pdbenergy.cli.__file__)\n"
            "except BaseException:\n"
            "    print('import     : FAILED')\n"
            "    traceback.print_exc()\n"
        )
        try:
            proc = subprocess.run(
                [self.python, "-c", probe], cwd=self.cwd, env=self.child_env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=180,
            )
        except Exception as exc:                              # pragma: no cover
            return [f"[诊断] 探针没能运行：{type(exc).__name__}: {exc}"]
        lines = ["[诊断] ---- 子进程自述（由界面代跑同一条路径）----"]
        lines += [f"[诊断] {ln}" for ln in (proc.stdout or "").splitlines()]
        lines.append(f"[诊断] 探针退出码 {proc.returncode}")
        return lines

    def check_import(self, module: str = "pdbenergy.cli") -> str | None:
        """Import ``module`` in a throwaway child; return ``None`` on success.

        Uses the same interpreter, cwd and environment the real jobs use, so a
        failure here is a failure every job would hit - which is exactly the kind
        of thing worth telling the user *before* they wait an hour for a run.
        """
        try:
            proc = subprocess.run(
                [self.python, "-c",
                 f"import importlib; importlib.import_module({module!r})"],
                cwd=self.cwd, env=self.child_env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=180,
            )
        except Exception as exc:                              # pragma: no cover
            return f"{type(exc).__name__}: {exc}"
        if proc.returncode == 0:
            return None
        tail = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        return tail[-1] if tail else f"退出码 {proc.returncode}"

    def import_hint(self, module: str, reason: str) -> str:
        """A diagnosis to print instead of a bare ``No module named ...``."""
        pkg = os.path.join(self.project_root, "pdbenergy")
        cli = os.path.join(pkg, "cli.py")
        lines = [f"[界面] 子进程无法导入 {module}：{reason}"]
        debugger = running_under_debugger()
        if debugger:
            # Listed first because it is, by a wide margin, the likeliest cause -
            # and the only one the user can fix in five seconds.
            lines += [
                f"[界面] ⚠ 界面正跑在调试器下（{debugger}）。",
                "[界面]   调试器会把运行时注入界面启动的每一个子进程，"
                "其中的 `python -m` 会以「No module named ...」静默失败（退出码 0）。",
                "[界面]   这不是环境坏了：同一个解释器 `-c \"import pdbenergy.cli\"` 是成功的。",
                r"[界面]   修法：不要用调试方式启动界面。用 scripts\start_gui.cmd、"
                "终端 `python -m pdbenergy.gui_qt`，"
                r"或在 .vscode/launch.json 里给该配置加上 \"noDebug\": true。",
            ]
        lines += [
            f"[界面]   解释器    : {self.python}"
            f"（存在：{os.path.isfile(self.python)}）",
            f"[界面]   项目根目录: {self.project_root}"
            f"（存在：{os.path.isdir(self.project_root)}）",
            f"[界面]   包目录    : {pkg}（存在：{os.path.isdir(pkg)}）",
            f"[界面]   cli.py    : {cli}（存在：{os.path.isfile(cli)}）",
            f"[界面]   工作目录  : {self.cwd}",
            "[界面] 传给子进程的 Python 环境变量：",
            *self.python_env_report(),
            "[界面] 请在终端手跑这条，它和界面走的是同一个解释器：",
            f'[界面]   {self.python} -c "import pdbenergy.cli; print(pdbenergy.cli.__file__)"',
            "[界面] 若上面成功而界面仍失败：先看上面那条调试器警告，"
            "否则就是界面进程继承了坏掉的 PYTHONHOME/PYTHONPATH，"
            "关掉界面、用干净环境重新启动。",
            f"[界面] 若包目录或 cli.py 显示不存在，说明界面用的是另一个副本，"
            f"请在 {self.project_root} 下启动。",
        ]
        return "\n".join(lines)

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

        env = self.child_env()
        # Merge stderr into stdout so the log reads in causal order.
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        job.module = module
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
                reason = self._import_failure(job)
                if reason:
                    # The whole log, not just the offending line: a bare
                    # "No module named X" is unactionable, and the lines above
                    # it often carry the real story.
                    extra = [f"[界面] 子进程退出码 {code}；完整输出见上。",
                             self.import_hint(job.module, reason)]
                    extra += self.diagnose_child(job.module)
                    job.lines.extend(extra)
                    job.produced += len(extra)
                # Status is written *last*.  Readers stop polling as soon as they
                # see a terminal status, so setting it before appending would
                # make them miss the diagnosis entirely.
                #
                # A child that never managed to import the module can still exit
                # 0 in some setups, so a clean exit code alone is not enough to
                # call the job successful - that is what let a run that produced
                # nothing at all report "任务完成".
                job.status = "failed" if (code != 0 or reason) else "done"
            try:                                   # release the pipe promptly
                if job.process.stdout is not None:
                    job.process.stdout.close()
            except Exception:
                pass

    @staticmethod
    def _import_failure(job: Job) -> str | None:
        """The child's own line proving it could not import what we asked for."""
        for line in job.lines:
            if any(marker in line for marker in IMPORT_FAILURE_MARKERS):
                return line.strip()
        return None

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
        # jsonutil strips NaN/Infinity: Python emits them as bare tokens, which
        # are not valid JSON, so the browser fails with
        # "Unexpected token 'N', ... is not valid JSON".
        from .jsonutil import dumps_json

        body = dumps_json(payload).encode("utf-8")
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

        The mapping itself lives in :mod:`pdbenergy.actions` so the desktop GUI
        and this one cannot drift apart; here we only translate its errors into
        an HTTP status and hand the command to the job manager.
        """
        from .actions import ActionError, build_job

        if not body:
            return self._json({"error": "empty request"}, 400)
        try:
            label, module, args = build_job(
                body.get("action"), body, outputs_dir=self.dirs["outputs"]
            )
        except ActionError as exc:
            return self._json({"error": str(exc)}, 400)
        return self._ok(label, module, args)

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
<title>PDBEnergy — 训练 / 预测 / 测试</title>
<style>
:root{
  --bg:#0f1419; --panel:#161c24; --panel2:#1d242e; --line:#2a3441;
  --fg:#dde3ea; --dim:#8b98a8; --accent:#4da3ff; --ok:#3fb950; --warn:#d29922;
  --bad:#f85149;
  /* 全站字体统一为 Times New Roman；中文回退到宋体（与 Times 同属衬线体） */
  --font:"Times New Roman", Times, "SimSun", "宋体", "Songti SC", serif;
  --mono:var(--font);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 var(--font)}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;
  background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}
header h1{font-size:22px;margin:0;font-weight:700;letter-spacing:.5px;
  font-family:var(--font)}
header .spacer{flex:1}
.pill{font:13px/1 var(--font);padding:5px 9px;border-radius:999px;
  background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
.pill b{color:var(--fg);font-weight:700}
main{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 52px)}
nav{padding:14px 10px;border-right:1px solid var(--line);background:var(--panel)}
nav button{display:block;width:100%;text-align:left;margin-bottom:4px;padding:9px 12px;
  background:transparent;border:1px solid transparent;border-radius:7px;color:var(--dim);
  font:15px var(--font);cursor:pointer}
nav button:hover{background:var(--panel2);color:var(--fg)}
nav button.on{background:var(--panel2);border-color:var(--accent);color:var(--fg)}
section{padding:18px 22px;display:none}
section.on{display:block}
h2{font-size:19px;margin:0 0 4px;font-weight:700}
h3{font-size:15px;margin:22px 0 8px;color:var(--dim);font-weight:700;
  letter-spacing:.4px}
p.hint{color:var(--dim);font-size:13.5px;margin:2px 0 14px;max-width:88ch}
code{font-family:var(--font);background:#0a0e12;border:1px solid var(--line);
  border-radius:4px;padding:0 4px;font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:14px 16px;margin-bottom:14px}
.grid{display:grid;gap:10px 14px;grid-template-columns:repeat(auto-fill,minmax(200px,1fr))}
label{display:block;font-size:13.5px;color:var(--dim);margin-bottom:4px}
input,select{width:100%;padding:7px 9px;background:var(--bg);color:var(--fg);
  border:1px solid var(--line);border-radius:6px;font:14px var(--font)}
input[type=checkbox]{width:auto}
button.act{padding:8px 15px;border-radius:7px;border:1px solid var(--accent);
  background:#153055;color:#cfe6ff;font:14px var(--font);cursor:pointer}
button.act:hover{background:#1b3d69}
button.act.ghost{border-color:var(--line);background:var(--panel2);color:var(--fg)}
button.act.danger{border-color:var(--bad);background:#3a1a1a;color:#ffd7d5}
button.act:disabled{opacity:.45;cursor:not-allowed}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font:13.5px var(--font)}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:700;position:sticky;top:0;background:var(--panel)}
tbody tr:hover{background:var(--panel2)}
.scroll{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:7px}
#log{font:13px/1.5 var(--font);background:#0a0e12;border:1px solid var(--line);
  border-radius:7px;padding:10px;height:230px;overflow:auto;white-space:pre-wrap;
  word-break:break-word}
#logwrap{padding:0 22px 20px}
.badge{display:inline-block;padding:2px 7px;border-radius:5px;font-size:12px;
  font-family:var(--font)}
.b-ok{background:#12351c;color:var(--ok)} .b-bad{background:#3a1a1a;color:var(--bad)}
.b-run{background:#153055;color:var(--accent)} .b-idle{background:var(--panel2);color:var(--dim)}
.figs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.figs figure{margin:0;background:var(--panel2);border:1px solid var(--line);
  border-radius:7px;padding:8px}
.figs img{width:100%;border-radius:5px;background:#fff}
.figs figcaption{font:12px var(--font);color:var(--dim);margin-top:6px;text-align:center}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font:13.5px var(--font)}
.kv div:nth-child(odd){color:var(--dim)}
.warn{color:var(--warn)} .bad{color:var(--bad)} .ok{color:var(--ok)}

/* ---- 名词注解：悬停显示解释 ---- */
[data-tip]{border-bottom:1px dotted var(--accent);cursor:help;position:relative}
[data-tip]:hover::after{
  content:attr(data-tip);
  position:absolute;left:0;top:135%;z-index:80;
  min-width:240px;max-width:min(420px,72vw);
  background:#070a0d;color:var(--fg);border:1px solid var(--accent);
  border-radius:7px;padding:8px 11px;font-size:13.5px;line-height:1.6;
  white-space:normal;text-align:left;font-weight:400;
  box-shadow:0 8px 26px rgba(0,0,0,.65);pointer-events:none;
}
th[data-tip]:hover::after{position:fixed}

/* ---- 帮助页 ---- */
details{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  margin-bottom:8px;padding:0}
details[open]{border-color:var(--accent)}
summary{cursor:pointer;padding:11px 14px;font-size:15px;font-weight:700;
  list-style:none;display:flex;gap:8px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary::before{content:"＋";color:var(--accent);font-weight:700}
details[open] summary::before{content:"－"}
summary:hover{color:var(--accent)}
details .body{padding:0 16px 14px 34px;color:var(--fg);
  border-top:1px solid var(--line);margin-top:0;padding-top:12px}
details .body p{margin:0 0 9px}
details .body ul{margin:0 0 9px;padding-left:20px}
details .body li{margin-bottom:5px}
details .body table{margin:8px 0;font-size:13px}
details .body th{position:static;background:transparent}
.gloss{display:grid;gap:6px 18px;grid-template-columns:auto 1fr;font-size:13.5px}
.gloss dt{color:var(--accent);font-weight:700;white-space:nowrap}
.gloss dd{margin:0 0 6px}
</style>
</head>
<body>
<header>
  <h1>PDBEnergy</h1>
  <span class="pill" data-tip="data/raw 目录里的 PDB 结构文件数量">原始结构 <b id="s-raw">–</b></span>
  <span class="pill" data-tip="已经用力场算好能量的构象系综数量（data/interim/*.npz）">已标注 <b id="s-lab">–</b></span>
  <span class="pill" data-tip="原始结构数 − 已标注数，即还需处理的条目">待标注 <b id="s-pend">–</b></span>
  <span class="pill" data-tip="体检结果：扫描了多少条目、其中多少值得打标签">体检 <b id="s-inv">–</b></span>
  <span class="spacer"></span>
  <span class="pill" id="s-job">空闲</span>
  <button class="act ghost" id="btn-cancel" disabled
          data-tip="终止当前任务。任务跑在子进程里，取消是真的结束进程">取消任务</button>
  <button class="act ghost" onclick="refresh()"
          data-tip="重新读取项目状态（每 15 秒也会自动刷新一次）">刷新</button>
</header>
<main>
<nav>
  <button data-tab="overview" class="on">概览</button>
  <button data-tab="data">数据 / 打标签</button>
  <button data-tab="train">训练</button>
  <button data-tab="eval">评估 / 测试</button>
  <button data-tab="predict">预测</button>
  <button data-tab="iterate">迭代训练</button>
  <button data-tab="help">帮助 / 常见问题</button>
</nav>
<div>
  <section id="tab-overview" class="on">
    <h2>概览</h2>
    <p class="hint">项目当前状态。所有面板共享底部同一个<span data-tip="当前任务的实时标准输出。界面显示的命令行可以直接复制到终端重跑">实时日志</span>。</p>
    <div class="card"><div class="kv" id="ov-kv"></div></div>
    <h3>已有运行</h3>
    <div class="scroll"><table id="ov-runs"></table></div>
  </section>

  <section id="tab-data">
    <h2>数据 / 打标签</h2>
    <p class="hint"><b>扫描</b>会解析 data/raw 里每个文件（几千个文件要几分钟），
      过滤掉非蛋白、过短和过长的条目，并告诉你哪些值得花 OpenMM 时间。
      <b>打标签</b>是最慢的一步，实测每个蛋白 2–4 分钟；已完成的会跳过，可以分批做。</p>
    <div class="card">
      <div class="grid">
        <div><label><span data-tip="只处理不超过这个残基数的条目。大蛋白的构建代价超线性增长（9000 残基的条目会让 OpenMM 几乎跑不完）">最大残基数</span>
          <code>max_residues</code></label><input id="d-max" type="number" value="120"></div>
        <div><label><span data-tip="过滤掉单残基、二肽这类没有构象能可学的小碎片">最小残基数</span>
          <code>min_residues</code></label><input id="d-min" type="number" value="10"></div>
        <div><label><span data-tip="OpenMM 内部使用的并行线程数。实测 8 线程比 1 线程快约 3.5–4 倍">OpenMM 线程数</span></label>
          <input id="d-threads" type="number" value="8"></div>
        <div><label><span data-tip="同时处理多少个蛋白（多进程）。workers × threads 不要超过物理核数，否则互相抢核反而更慢">并行进程数</span>
          <code>workers</code></label><input id="d-workers" type="number" value="1"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" data-tip="解析 data/raw 全部文件并生成体检报告，结果决定后续能标注哪些条目"
                onclick="post('scan',{max_residues:+d('d-max'),min_residues:+d('d-min')})">扫描 data/raw</button>
        <button class="act" data-tip="按当前的残基上下限，列出还没标注、且值得标注的条目"
                onclick="loadPending()">列出待标注</button>
        <button class="act ghost" data-tip="从 RCSB 下载内置的那批小蛋白条目（需要联网）"
                onclick="post('download',{})">下载内置条目</button>
        <button class="act ghost" data-tip="按蛋白质切分训练/验证/测试集，并计算目标的均值与标准差"
                onclick="post('dataset',{})">构建数据集切分</button>
        <button class="act ghost" data-tip="由体检报告推算：不同残基上限下要标多少条、多少构象、多少磁盘、多长时间"
                onclick="post('estimate_workload',{})">工作量估算</button>
      </div>
      <p class="hint" style="margin:12px 0 0">提示：workers × threads 不要超过物理核数
        （4 核 8 线程的机器建议 <code>4 × 2</code>）。</p>
    </div>
    <h3>条目 <span id="d-count" class="badge b-idle">未加载</span></h3>
    <div class="row" style="margin-bottom:8px">
      <button class="act ghost" onclick="selectAll(true)">全选</button>
      <button class="act ghost" onclick="selectAll(false)">清空</button>
      <input id="d-limit" type="number" value="50" style="width:90px"
             data-tip="本次最多标注多少条。剩余的下次继续，已完成的会自动跳过">
      <button class="act" data-tip="对勾选的条目启动力场打标签（会跑 OpenMM，耗时最长）"
              onclick="labelling()">标注已勾选</button>
    </div>
    <div class="scroll"><table id="d-table"></table></div>
  </section>

  <section id="tab-train">
    <h2>训练</h2>
    <p class="hint">超参改动会写进命令行并显示在日志里，所以每次运行都可手工复现。
      改 <code>hidden_dim</code>/<code>cutoff</code>/<code>n_rbf</code>
      会使<span data-tip="把「结构→图」的结果缓存下来复用。约 0.5 MB/构象，2 万构象约 10 GB">图缓存</span>失效并改变权重形状——那种情况下不能热启动。</p>
    <div class="card">
      <div class="grid">
        <div><label><span data-tip="SchNet：消息传递图神经网络，直接吃 3D 结构。MLP：手工描述符基线，用 Rg、接触数等 24 维全局量">模型</span></label>
          <select id="t-model">
          <option value="schnet">SchNet（图神经网络）</option>
          <option value="mlp">MLP（手工描述符基线）</option></select></div>
        <div><label><span data-tip="本次训练把训练集完整过几遍。配置里的 epochs 永远是「这次再跑多少轮」，续训时不会重复已跑的轮次">训练轮数</span>
          <code>epochs</code></label><input id="t-epochs" type="number" value="25"></div>
        <div><label><span data-tip="一次梯度更新用多少个构象。大 batch 的矩阵运算效率更高（实测每样本耗时从 2.13 s 降到 1.36 s），但更占内存">批大小</span>
          <code>batch_size</code></label><input id="t-batch" type="number" value="16"></div>
        <div><label><span data-tip="AdamW 的步长。太大不收敛，太小训练慢">学习率</span>
          <code>learning_rate</code></label><input id="t-lr" type="number" step="0.0001" value="0.0005"></div>
        <div><label><span data-tip="每个原子的特征向量维度。越大容量越强、越慢，也越容易在小数据集上过拟合">隐藏维度</span>
          <code>hidden_dim</code></label><input id="t-hidden" type="number" value="64"></div>
        <div><label><span data-tip="消息传递层数。一个原子能「看」到约 层数 × cutoff 的范围（本项目 3 × 4 Å = 12 Å）">交互层数</span>
          <code>n_interactions</code></label><input id="t-int" type="number" value="3"></div>
        <div><label><span data-tip="按蛋白质切分：测试集蛋白训练中从未出现，是唯一诚实的做法。按帧切分：同一蛋白的近似重复构象会同时出现在训练和测试里，指标会被严重高估">数据切分方式</span></label>
          <select id="t-split">
          <option value="protein">按蛋白质（正确）</option>
          <option value="frame">按帧（数据泄漏，仅用于对照）</option></select></div>
        <div><label><span data-tip="输出目录名，结果写到 outputs/&lt;tag&gt;/">输出目录名</span>
          <code>tag</code></label><input id="t-tag" value="schnet_protein"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="t-nocache">
          <span data-tip="不把图缓存进内存，改成每次现场构造。数据集超过约 2 万构象时必须打开（缓存会 OOM）">禁用图缓存</span></label>
        <label style="margin:0"><input type="checkbox" id="t-recompute">
          <span data-tip="热启动/续训时默认沿用检查点里的目标均值与标准差。如果新蛋白的分布确实不同，才需要重算">重算归一化</span></label>
      </div>
      <div class="grid" style="margin-top:12px">
        <div><label><span data-tip="用旧模型的权重初始化，但优化器、轮次计数、历史全部重来。数据变多时用这个">&nbsp;热启动</span>
          <code>--init-from</code></label><input id="t-init" placeholder="outputs/.../checkpoint.pt"></div>
        <div><label><span data-tip="连 AdamW 的两个动量和轮次计数一起恢复，接着往下跑。长时间训练被中断后用它">&nbsp;续训</span>
          <code>--resume</code></label><input id="t-resume" placeholder="outputs/某个运行目录"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" data-tip="启动训练。任务在子进程里跑，界面不会卡，日志实时回传"
                onclick="train()">开始训练</button>
        <span class="hint" style="margin:0">热启动与续训两者只能填一个</span>
      </div>
    </div>
    <div id="t-recent"></div>
  </section>

  <section id="tab-eval">
    <h2>评估 / 测试</h2>
    <p class="hint">选一个运行做评估并看图。注意：<b>MAE 单独看会骗人</b>——
      务必同时看<span data-tip="(预测最大值−最小值)/(真实最大值−最小值)。只有百分之几，说明模型基本在输出一个常数">&nbsp;预测跨度</span>和
      <span data-tip="同一个蛋白内部的排序相关性。这才是打分函数真正需要的指标">&nbsp;组内 ρ</span>。</p>
    <div class="card">
      <div class="grid">
        <div><label><span data-tip="outputs/ 下每个含 checkpoint.pt 的目录">运行</span></label>
          <select id="e-run"></select></div>
        <div><label><span data-tip="数据泄漏消融实验的训练轮数">消融轮数</span></label>
          <input id="e-epochs" type="number" value="20"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="act" data-tip="在验证集和测试集上算指标，并生成 parity、残差、学习曲线等图"
                onclick="evaluate()">评估该运行</button>
        <button class="act ghost" data-tip="用同样配置只改切分方式训练两次，量化「按帧切分」能把指标虚高多少"
                onclick="post('ablate',{epochs:+d('e-epochs'),model:'mlp'})">数据泄漏消融</button>
      </div>
    </div>
    <div class="card" id="e-metrics"></div>
    <div class="figs" id="e-figs"></div>
  </section>

  <section id="tab-predict">
    <h2>预测</h2>
    <p class="hint">给 PDB 文件打分并排序。输出的是<span data-tip="ΔE = E − min(E)，相对该蛋白自身最低构象能的差值。只在同一分子内部可比，跨分子无意义">&nbsp;相对构象能 ΔE</span>。
      <code>--verify</code> 会额外用 OpenMM 算一次真值能量，可以直接看到误差（慢一些）。</p>
    <div class="card">
      <div class="grid">
        <div><label><span data-tip="模型检查点：权重 + 模型/特征配置 + 归一化统计量。单个文件就足以做推理">检查点</span>
          <code>checkpoint</code></label><select id="p-ckpt"></select></div>
        <div><label><span data-tip="NMR 条目一个文件里有几十个模型，这里限制每个文件取前几个">每个文件取前 N 个模型</span></label>
          <input id="p-models" type="number" value="1"></div>
        <div><label><span data-tip="verify 时 OpenMM 使用的线程数">OpenMM 线程数</span></label>
          <input id="p-threads" type="number" value="4"></div>
        <div><label><span data-tip="也可以直接手填路径，多个用英文逗号分隔">或手填文件路径</span></label>
          <input id="p-paths" placeholder="data/raw/1L2Y.pdb"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="p-verify">
          <span data-tip="额外用物理力场算一次真值能量并显示误差。这是检验模型是否可用的唯一诚实方式">&nbsp;对照真值（--verify）</span></label>
        <button class="act" data-tip="对选中的 PDB 文件逐个预测并按预测能量排序" onclick="predict()">开始预测</button>
        <button class="act ghost" onclick="loadPredictions()">读取上次结果</button>
      </div>
      <div class="grid" style="margin-top:12px">
        <div><label>从 data/raw 选择文件（可多选）</label>
          <select id="p-file" size="6" multiple></select></div>
      </div>
    </div>
    <div class="scroll"><table id="p-table"></table></div>
  </section>

  <section id="tab-iterate">
    <h2>迭代训练</h2>
    <p class="hint">每轮：标注新条目 → 重建数据集 → 从上一轮热启动 → 评估 → 记录。
      <span data-tip="第 1 轮把 train/val/test 的蛋白固定下来写进 split.json，之后新增的蛋白只进 train。否则测试集每轮都在变，「指标变好」可能只是难蛋白被换出去了">&nbsp;切分在第 1 轮冻结</span>，
      所以跨轮指标可比。<code>build-limit</code> 用来分批，避免一轮就把几百条全标了。</p>
    <div class="card">
      <div class="grid">
        <div><label><span data-tip="跑几轮「加数据→重训→评估→记录」">轮数</span>
          <code>rounds</code></label><input id="i-rounds" type="number" value="1"></div>
        <div><label><span data-tip="本轮最多标注多少条新条目。填 0 表示全部（几百条会跑几十小时）">本轮最多标注 N 条</span></label>
          <input id="i-limit" type="number" value="20"></div>
        <div><label><span data-tip="本轮允许标注的条目残基上限">最大残基数</span>
          <code>max_residues</code></label><input id="i-max" type="number" value="200"></div>
        <div><label>每轮训练轮数</label><input id="i-epochs" type="number" value="20"></div>
        <div><label>模型</label><select id="i-model">
          <option value="schnet">SchNet</option><option value="mlp">MLP</option></select></div>
        <div><label>OpenMM 线程数</label><input id="i-threads" type="number" value="8"></div>
        <div><label>并行进程数</label><input id="i-workers" type="number" value="1"></div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="margin:0"><input type="checkbox" id="i-build" checked>
          <span data-tip="自动发现 data/raw 里还没标注的条目并打标签">&nbsp;标注新条目</span></label>
        <label style="margin:0"><input type="checkbox" id="i-warm" checked>
          <span data-tip="每轮从上一轮检查点热启动。关掉就是每轮从零训，可作为对照">&nbsp;热启动</span></label>
        <button class="act" data-tip="启动多轮迭代训练" onclick="iterate()">运行迭代</button>
        <button class="act ghost" onclick="loadRounds()">读取轮次表</button>
      </div>
    </div>
    <div id="i-table"></div>
  </section>

  <section id="tab-help">
    <h2>帮助 / 常见问题</h2>
    <p class="hint">这里回答使用中最常遇到的疑问，以及这个项目<b>真实的能力边界</b>——
      有些问题的答案是「现在还做不到」，那也如实写在这里。</p>

    <h3>一、这个项目在做什么</h3>
    <details open><summary>这个模型到底预测什么？</summary><div class="body">
      <p>输入一个蛋白质构象（PDB 坐标 + 元素类型），输出它的<b>相对构象能</b>
        ΔE = E − min(E)，单位 kcal/mol。</p>
      <p>「相对」很关键：绝对能量主要由原子组成和长程溶剂化决定，换一个蛋白就完全不可比；
        真正有意义的是「这个构象比该蛋白自己最舒服的构象差多少」。</p>
      <p>标签（真值）不是实验数据，而是 <b>AMBER14 + GBn2 隐式溶剂</b>力场算出来的能量。
        所以模型学到的上限就是力场的上限——它是一台力场的快速代理，不是新的物理。</p>
    </div></details>

    <details><summary>训练好的模型，现在能当打分函数用吗？</summary><div class="body">
      <p><b>还不能。</b>在 14 个蛋白、730 个构象上实测：</p>
      <ul>
        <li>测试 MAE 121 kcal/mol，而「永远输出平均值」的常数基线是 107 kcal/mol——
          模型还不如常数预测器；</li>
        <li>预测跨度只有真实能量范围的 <b>2%</b>，说明它基本在输出一个常数；</li>
        <li>组内排序相关性 ρ 只有 0.17，也就是在同一蛋白内部不会排序。</li>
      </ul>
      <p>它确实带一点真实信号（全局 Spearman 0.43，在真实 NMR 条目上偶尔能排出正确顺序），
        但不足以用来挑构象。请把它当作<b>一条跑通的流水线</b>，而不是一个可用的工具。</p>
    </div></details>

    <h3>二、指标怎么读（最容易骗自己的部分）</h3>
    <details><summary>为什么说「MAE 单独看会骗人」？</summary><div class="body">
      <p>因为一个几乎只输出常数的模型，只要那个常数稍微好一点，MAE 就能「赢过常数基线」。
        本项目真实发生过：把径向基加密后，val MAE 从 129.5 降到 114.3（首次超过常数基线
        118.3），但 <b>Spearman 从 0.430 掉到 −0.036</b>，预测跨度仍只有 2%。</p>
      <p>它是靠「把预测压得更接近均值」赢的 MAE，代价是丢掉了排序能力。</p>
      <p><b>所以务必同时看三个量：</b>MAE（绝对误差）、预测跨度（是否在输出常数）、
        组内 ρ（能不能在同一个蛋白内部排序）。只看 MAE 会得出完全错误的结论。</p>
    </div></details>

    <details><summary>MAE / RMSE / R² / Spearman / 组内 ρ 分别是什么？</summary><div class="body">
      <table>
        <tr><th>指标</th><th>含义</th><th>陷阱</th></tr>
        <tr><td>MAE</td><td>平均绝对误差，kcal/mol</td><td>最直观，但对大误差不敏感</td></tr>
        <tr><td>RMSE</td><td>均方根误差</td><td>少数离群点就能拉高；与 MAE 差距大说明有离群点</td></tr>
        <tr><td>R²</td><td>解释了多少方差</td><td>会被「认出这是哪个蛋白」这种简单任务撑高</td></tr>
        <tr><td>Spearman ρ</td><td>秩相关（排序一致性）</td><td>全局值会被「区分不同蛋白」撑高</td></tr>
        <tr><td>组内 ρ</td><td>同一蛋白内部的排序相关性</td><td>这才是打分函数真正需要的</td></tr>
        <tr><td>预测跨度</td><td>(预测max−min)/(真实max−min)</td><td>只有百分之几 = 模型在输出常数</td></tr>
        <tr><td>常数基线</td><td>永远输出训练集平均值</td><td>任何模型都必须显著优于它</td></tr>
      </table>
    </div></details>

    <details><summary>为什么必须按蛋白质切分？「数据泄漏」指什么？</summary><div class="body">
      <p>同一个蛋白的相邻 MD 快照结构差异只有零点几埃、能量几乎一样。如果<b>按帧随机切分</b>，
        测试集里就会出现训练集的近似副本，模型只要「记住」就行，指标看起来很漂亮但什么都没学会。</p>
      <p>本项目直接量过这件事（不需要训练模型）：按帧切分时，<b>97.3% 的测试帧最近邻来自同一蛋白，
        52.7% 是近似重复</b>；按蛋白质切分时这两个数字都是 0%。</p>
      <p>所以默认用按蛋白质切分，测试集蛋白训练中从未出现。想自己看这个效果，用
        「评估 / 测试」页的<b>数据泄漏消融</b>按钮。</p>
    </div></details>

    <h3>三、数据与时间</h3>
    <details><summary>加更多数据能提升精度吗？</summary><div class="body">
      <p><b>本项目的实测答案是：不能（至少在当前设置下）。</b>在冻结切分下把训练集按
        25% / 50% / 100% 取子集：</p>
      <table>
        <tr><th>训练构象数</th><th>训练损失</th><th>验证 MAE</th></tr>
        <tr><td>81</td><td>0.3052</td><td>128.56</td></tr>
        <tr><td>160</td><td>0.3147</td><td><b>143.02</b></td></tr>
        <tr><td>323</td><td>0.3041</td><td>127.16</td></tr>
        <tr><td><i>常数基线</i></td><td>—</td><td><i>118.28</i></td></tr>
      </table>
      <p>数据翻倍验证 MAE 反而变差，三个点全在常数基线之上。而且<b>训练损失在任何数据量下
        都是 0.30–0.31</b>——模型总能拟合训练集，验证集纹丝不动。这是「学到的东西不迁移」，
        不是「样本不够」。加轮数也一样：4 轮和 20 轮结果相同。</p>
      <p>所以现阶段<b>不要</b>花几十小时去标几百个蛋白。先把「目标设计」修好
        （见下面最后一条）。</p>
    </div></details>

    <details><summary>打标签和训练各要多久？</summary><div class="body">
      <p>实测（4 核 8 线程笔记本 CPU）：</p>
      <ul>
        <li><b>打标签</b>：每个蛋白 2–4 分钟（8 线程）。100 个蛋白 ≈ 几小时；
          用 <code>--workers 4 --threads 2</code> 约快 2.6 倍。</li>
        <li><b>扫描</b>：2741 个文件约 6 分钟。</li>
        <li><b>训练</b>：323 个构象、SchNet、batch 16，约 100 秒/轮。</li>
      </ul>
      <p>打标签和扫描都可以中断后重跑：已完成的条目会跳过。</p>
    </div></details>

    <details><summary>磁盘和内存要多少？</summary><div class="body">
      <ul>
        <li>已标注数据（data/interim）：约 <b>5 KB/构象</b>。415 个蛋白 ≈ 122 MB。</li>
        <li>图缓存：约 <b>0.5 MB/构象，且全部读进内存</b>。2 万构象 ≈ 10 GB。
          超过预算就在「训练」页勾上<b>禁用图缓存</b>。</li>
        <li>原始 PDB：2741 个文件约 1.66 GB。</li>
      </ul>
      <p>用「工作量估算」按钮可以按残基上限推算这三项。</p>
    </div></details>

    <h3>四、操作与排错</h3>
    <details><summary>threads 和 workers 怎么配？</summary><div class="body">
      <p><b>threads</b> 是 OpenMM 内部线程数；<b>workers</b> 是同时处理几个蛋白（多进程）。</p>
      <p>关键在于 <code>workers × threads</code> <b>不要超过物理核数</b>，否则会互相抢核，
        反而比单进程更慢。4 核 8 线程的机器建议 <code>--workers 4 --threads 2</code>。</p>
      <p>OpenMM 的 CPU 并行收益是递减的：实测 8 线程只比 1 线程快约 3.5–4 倍。</p>
    </div></details>

    <details><summary>为什么改了模型结构就不能热启动？</summary><div class="body">
      <p><code>hidden_dim</code>、<code>n_interactions</code>、<code>cutoff</code>、
        <code>n_rbf</code> 里任何一个变了，权重张量的形状就对不上。</p>
      <p>最坏的结果不是报错，而是<b>只加载了一部分张量、剩下的保持随机</b>——
        得到一个看起来能跑、实际半随机初始化的模型。所以本项目在加载前逐字段比对配置，
        发现不一致就直接报错并指出是哪个字段变了。</p>
      <p>想要更大的模型，就换个输出目录从头训练（数据够了之后大模型才开始值钱）。</p>
    </div></details>

    <details><summary>热启动和续训有什么区别？</summary><div class="body">
      <table>
        <tr><th>做法</th><th>权重</th><th>优化器动量</th><th>轮次计数</th><th>用在哪</th></tr>
        <tr><td>从头训练</td><td>随机</td><td>全新</td><td>从 1 开始</td><td>换了结构或目标</td></tr>
        <tr><td>热启动</td><td>从检查点</td><td>全新</td><td>从 1 开始</td><td>数据变多，提升旧模型</td></tr>
        <tr><td>续训</td><td>从检查点</td><td><b>恢复</b></td><td><b>接着数</b></td><td>长训练被中断</td></tr>
      </table>
      <p>热启动默认<b>沿用检查点里的目标归一化</b>（均值/标准差）。换个尺度会让输出层一开始
        就标定错，前几轮全花在把尺度掰回来。只有新蛋白分布确实不同时，才勾「重算归一化」。</p>
    </div></details>

    <details><summary>「对照真值」/ verify 是做什么的？</summary><div class="body">
      <p>勾上之后，除了模型预测，还会用 OpenMM 把同一批结构再算一遍真实力场能量，
        于是你能直接看到<b>误差</b>，而不是只能相信预测值。</p>
      <p>这是检验模型是否可用的唯一诚实方式，代价是慢（要建 OpenMM 体系）。</p>
      <p><b>排序不对是正常的</b>：模型组内 ρ 只有 0.17，所以它排出错误顺序并不奇怪。</p>
    </div></details>

    <details><summary>怎么复现某一次运行？</summary><div class="body">
      <p>界面只负责拼命令行参数，逻辑全在 CLI 里。所以<b>日志里显示的那行命令可以直接复制到
        终端重跑</b>，结果一致。例如：</p>
      <p><code>python -m pdbenergy.cli train --model schnet --epochs 25 --tag schnet_protein</code></p>
      <p>每个运行目录里都保存了 <code>config.json</code>（用到的完整配置）、
        <code>history.json</code>（学习曲线）、<code>data_summary.json</code>（数据切分），
        检查点里还带着归一化统计量和切分清单——单个 <code>checkpoint.pt</code> 就足以做推理。</p>
    </div></details>

    <details><summary>报错 No module named 'pdbenergy' 怎么办？</summary><div class="body">
      <p>说明包不在 Python 的搜索路径里。界面已经给每个子任务设置了
        <code>PYTHONPATH</code>，所以从界面启动的任务一般不会遇到；如果你在终端手工跑命令遇到，
        两种解法：</p>
      <ul>
        <li>在<b>项目根目录</b>下运行（当前目录会在搜索路径里）；</li>
        <li>或者重装：<code>pip install -e .</code></li>
      </ul>
      <p>注意：如果项目目录被<b>改名或移动</b>过，editable 安装会静默失效
        （生成的路径钩子仍指向旧位置），此时必须重装。</p>
    </div></details>

    <details><summary>提示缺少 OpenMM / PDBFixer？</summary><div class="body">
      <p>只有涉及力场的功能需要它们：<b>打标签、预测（因为要加氢）、verify</b>。
        训练和评估不需要。</p>
      <p>安装：<code>pip install openmm pdbfixer</code> 或
        <code>pip install -e ".[physics]"</code>。</p>
      <p>为什么预测也要 OpenMM？因为 PDB 文件通常不含氢，而力场是全原子的、
        训练标签也是全原子的，所以推理前必须先把氢补上。</p>
    </div></details>

    <details><summary>能不能用 GPU？</summary><div class="body">
      <p>训练代码支持（<code>--device cuda</code>），本项目所有结果都是 CPU 上跑的
        （4 核笔记本）。OpenMM 部分也有 CUDA 平台，但对小蛋白来说 CPU 反而更划算——
        体系太小，GPU 的传输开销占主导。</p>
    </div></details>

    <h3>五、术语表</h3>
    <div class="card"><dl class="gloss">
      <dt>构象</dt><dd>不改变化学键、只绕单键旋转得到的分子形状。</dd>
      <dt>残基</dt><dd>蛋白质链上的一个氨基酸单元。46 个残基的蛋白约 640 个原子（含氢）。</dd>
      <dt>打标签</dt><dd>用物理力场算出每个构象的能量，作为神经网络的学习目标。</dd>
      <dt>力场</dt><dd>用经验函数近似能量随原子坐标变化的模型。本项目用 AMBER14 + GBn2 隐式溶剂。</dd>
      <dt>隐式溶剂</dt><dd>把水当成连续介质，而不是显式加几千个水分子。GBn2 是其中一种近似。</dd>
      <dt>二面角旋转</dt><dd>绕可旋转单键旋转一侧子树。精确保持所有键长键角，是生成构象的正确方式。</dd>
      <dt>能量最小化</dt><dd>求能量的局部极小值（L-BFGS）。用来定义相对能量的零点。</dd>
      <dt>能量均分</dt><dd>温度 T 下势能平均值比极小值高约 ½N·kT。640 原子蛋白在 300 K 约高 570 kcal/mol。</dd>
      <dt>图神经网络</dt><dd>用「原子=节点、邻近关系=边」的图表示分子，通过消息传递更新每个原子的特征。</dd>
      <dt>消息传递</dt><dd>每个原子收集邻居信息并更新自己，叠 k 层就传播 k 跳。</dd>
      <dt>SchNet</dt><dd>一种消息传递网络，边的权重由原子间距离经 RBF 展开后决定。</dd>
      <dt>径向基（RBF）</dt><dd>把标量距离展开成一组高斯函数值，让网络更容易学。本项目间距 0.161 Å。</dd>
      <dt>cutoff</dt><dd>只让距离小于它的原子对交换消息。越小越快，但会截断长程物理。</dd>
      <dt>求和池化</dt><dd>把每个原子的能量贡献相加得到总能量。保证能量是广延量。</dd>
      <dt>归一化</dt><dd>目标减均值除标准差，让损失尺度稳定。必须用训练集统计量，且存进检查点。</dd>
      <dt>Huber 损失</dt><dd>误差小时二次、大时线性。防止少数高能离群点主导梯度。</dd>
      <dt>AdamW</dt><dd>带一阶/二阶动量并解耦权重衰减的优化器。</dd>
      <dt>早停</dt><dd>验证损失连续若干轮不改善就停下来，并恢复最佳轮次的权重。</dd>
      <dt>数据泄漏</dt><dd>测试集里含有训练集的近似副本，导致指标虚高。本项目实测按帧切分泄漏率 53%。</dd>
      <dt>消融实验</dt><dd>只改一个变量、其余完全相同的对照实验。</dd>
      <dt>检查点</dt><dd>权重 + 模型/特征配置 + 归一化统计量 + 切分清单。自解释，可单独做推理。</dd>
      <dt>图缓存</dt><dd>把「结构→图」的结果存内存复用。约 0.5 MB/构象，是内存的主要消耗。</dd>
    </dl></div>

    <h3>六、下一步该改什么</h3>
    <details><summary>想让精度真正提升，应该从哪里入手？</summary><div class="body">
      <p>按实测证据排序（完整论述见 <code>TeachFlow.md</code> §12.3）：</p>
      <ul>
        <li><b>① 目标设计（最该动的地方）。</b>现在的 ΔE 把两件性质不同的东西混在一起：
          热激发幅度（MD 帧比极小值高 300–1000 kcal/mol，且随蛋白大小线性增长）
          和构象形变（二面角帧通常只有 0–100 kcal/mol）。模型要从亚 0.1 Å 的几何细节
          同时预测这两者，动态范围跨两个数量级。可以试：按原子数归一的目标 ΔE/N、
          砍掉 450 K 那条高温尾巴、或者把两部分分开建模。</li>
        <li><b>② 特征分辨率。</b>基函数间距 0.161 Å 比键长热涨落（0.05–0.1 Å）还粗，
          加密到 64–96 后 MAE 确实降了 12–16%——但代价是排序能力塌掉。
          它有用，但不是全部答案。</li>
        <li><b>③ 排序损失。</b>用 pairwise ranking loss 直接优化组内 ρ，
          因为那才是打分函数的实际用途。</li>
        <li><b>④ 力监督。</b>把能量梯度也加进损失，这是机器学习势函数的标准做法，
          对局部形变极其敏感。</li>
        <li><b>⑤ 最后才是加数据。</b>等目标修好、指标越过常数基线之后，
          数据量大概率会重新变成瓶颈。</li>
      </ul>
      <p>复现判定这些结论的实验：<code>scripts/learning_curve.py</code>（数据量）、
        <code>scripts/compare_configs.py</code>（配置消融）。</p>
    </div></details>
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
const CN = {val:'验证', test:'测试'};

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
      ? `${STATE.inventory.usable}/${STATE.inventory.scanned} 可用` : '未扫描';
    renderOverview();
    fillRuns();
    fillCheckpoints();
  } catch (e) { log('刷新状态失败：' + e.message); }
}

function renderOverview() {
  if (!STATE) return;
  const c = STATE.counts, inv = STATE.inventory;
  const rows = [
    ['工作目录', STATE.cwd],
    ['data/raw 里的 PDB 文件', c.raw_pdb],
    ['已打标签（data/interim）', c.labelled],
    ['待标注', c.pending],
    ['体检结果', inv ? `扫描 ${inv.scanned} 条，可用 ${inv.usable}，排除 ${inv.unusable}` : '未扫描'],
    ['可用条目的残基范围', inv ? `${inv.min_residues} – ${inv.max_residues}` : '—'],
  ];
  $('ov-kv').innerHTML = rows.map(([k,v]) =>
    `<div>${esc(k)}</div><div>${esc(v)}</div>`).join('');

  const runs = STATE.runs || [];
  let h = `<thead><tr><th>运行</th>
    <th title="切分方式 / 预测目标">配置</th>
    <th title="参与训练的蛋白数">蛋白</th>
    <th title="总构象数">构象</th>
    <th title="验证集平均绝对误差，kcal/mol">验证 MAE</th>
    <th title="测试集平均绝对误差，kcal/mol">测试 MAE</th>
    <th title="测试集 Spearman 秩相关">测试 ρ</th>
    <th title="该运行是否已经评估过">状态</th></tr></thead><tbody>`;
  if (!runs.length) h += `<tr><td colspan="8" style="color:var(--dim)">还没有训练好的运行</td></tr>`;
  for (const r of runs) {
    const m = (r.metrics || {}), v = m.val || {}, t = m.test || {};
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
    h += `<div>${CN[s]} MAE / RMSE</div><div>${num(x.mae)} / ${num(x.rmse)} ${verdict}</div>`;
    h += `<div>${CN[s]} R² / Spearman</div><div>${num(x.r2,4)} / ${num(x.spearman,4)}</div>`;
    h += `<div>${CN[s]}组内平均 ρ</div><div>${num(x.rank_rho,4)} ${x.rank_rho!=null && x.rank_rho<0.3 ? '<span class="warn">← 排序能力弱</span>':''}</div>`;
    if (b != null) h += `<div>${CN[s]}常数基线 MAE</div><div>${num(b)}</div>`;
  }
  h += '</div><p class="hint" style="margin:12px 0 0">预测跨度不在 metrics.json 里，'+
       '看 parity 图、或跑 <code>scripts/compare_configs.py</code> 的输出来确认模型是否在输出常数。'+
       '详见「帮助」页。</p>';
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

/* ---- 数据面板 ---- */
async function loadPending() {
  const q = `?pending=1&max_residues=${d('d-max')}&min_residues=${d('d-min')}&limit=300`;
  try {
    const r = await api('/api/inventory' + q);
    renderEntries(r.rows, `待标注 ${r.n_rows}`);
  } catch (e) {
    log('读取体检报告失败：' + e.message + ' —— 请先点「扫描 data/raw」');
  }
}
function renderEntries(rows, label) {
  $('d-count').textContent = label;
  $('d-count').className = 'badge b-run';
  let h = `<thead><tr><th>选择</th><th>PDB 编号</th>
    <th title="标准氨基酸残基数">残基数</th>
    <th title="文件里的原子总数">原子数</th>
    <th title="NMR 条目的模型个数，多模型是白送的实验构象">模型数</th>
    <th title="EXPDTA：X-RAY / SOLUTION NMR 等">实验方法</th>
    <th title="被排除的原因">备注</th></tr></thead><tbody>`;
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
  if (limit > 0 && ids.length > limit) log(`只取前 ${limit} 条（可在输入框调整）`);
  post('label', {ids: chosen, threads: +d('d-threads'), workers: +d('d-workers')});
}

/* ---- 训练面板 ---- */
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
  if (!ck) return log('没有可用的检查点，请先训练');
  post('predict', {paths, checkpoint: ck, max_models: +d('p-models'),
                   threads: +d('p-threads'), verify: $('p-verify').checked});
}
async function fillCheckpoints() {
  try {
    const r = await api('/api/checkpoints');
    const sel = $('p-ckpt'), cur = sel.value;
    sel.innerHTML = r.checkpoints.map(c =>
      `<option value="${esc(c.path)}">${esc(c.name)}</option>`).join('') ||
      '<option value="">（无可用的检查点，请先训练）</option>';
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
    let h = `<thead><tr><th>文件</th>
      <th title="NMR 条目里的第几个模型">模型序号</th>
      <th title="模型预测的相对构象能 ΔE，kcal/mol">预测 ΔE</th>
      <th title="用 OpenMM 力场算出的真值 ΔE（需要勾选 verify）">真值 ΔE</th>
      <th title="预测 − 真值">误差</th>
      <th>残基数</th></tr></thead><tbody>`;
    for (const p of r.rows) {
      h += `<tr><td>${esc(p.name)}</td><td>${p.model_index}</td>
        <td>${num(p.predicted_relative_energy_kcal_per_mol,3)}</td>
        <td>${p.true_relative_energy_kcal_per_mol==null?'—':num(p.true_relative_energy_kcal_per_mol,3)}</td>
        <td>${p.error_kcal_per_mol==null?'—':num(p.error_kcal_per_mol,3)}</td>
        <td>${p.n_residues}</td></tr>`;
    }
    $('p-table').innerHTML = h + '</tbody>';
  } catch (e) { log('没有可读的预测结果（先跑一次预测）'); }
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
    let h = `<div class="scroll"><table><thead><tr><th>轮次</th><th>蛋白数</th>
      <th title="训练/验证/测试的构象数">训练/验证/测试</th>
      <th title="验证集 MAE，选模型只看它">验证 MAE</th>
      <th title="测试集 MAE，只用于报告">测试 MAE</th>
      <th title="测试集全局 Spearman">测试 ρ</th>
      <th title="同一蛋白内部的排序相关性">组内 ρ</th>
      <th title="永远输出训练集平均值的基线">常数基线</th>
      <th title="预测跨度占真实跨度的比例">跨度</th></tr></thead><tbody>`;
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
  } catch(e) { log('读取轮次表失败：' + e.message); }
}

/* ---- 任务与日志 ---- */
async function post(action, payload) {
  try {
    const r = await api('/api/job', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(Object.assign({action}, payload||{}))});
    startWatch(r.job);
  } catch (e) { log('启动任务失败：' + e.message); }
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
    if (j.dropped) $('log').textContent += `\n[界面] ${j.dropped} 行较早的输出已被丢弃\n`;
    if (j.lines.length) {
      const el = $('log');
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent += j.lines.join('\n') + '\n';
      if (atBottom) el.scrollTop = el.scrollHeight;
    }
    CURSOR = j.cursor;
    $('s-job').textContent = `${{running:'运行中',done:'已完成',failed:'失败',cancelled:'已取消'}[j.status]||j.status} ${j.elapsed.toFixed(0)} 秒`;
    if (j.status !== 'running') {
      clearInterval(POLL); POLL = null; ACTIVE = null;
      $('btn-cancel').disabled = true;
      $('log-label').textContent = {done:'已完成',failed:'失败',cancelled:'已取消'}[j.status]||j.status;
      $('log-label').className = 'badge ' + (j.status==='done'?'b-ok':'b-bad');
      refresh();
      if (j.status === 'done') { loadPredictions(); loadRounds(); }
    }
  } catch (e) { /* 轮询失败是暂时的，忽略 */ }
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
        "manager": JobManager(),
        "cfg": cfg,
        "dirs": {
            "raw": resolve_dir(args.raw_dir),
            "interim": resolve_dir(args.interim_dir),
            "processed": resolve_dir(args.processed_dir),
            "outputs": resolve_dir(args.outputs_dir),
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
