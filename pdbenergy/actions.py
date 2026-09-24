"""Translate a UI action into a CLI invocation.

Both interfaces - the PySide6 desktop app (:mod:`pdbenergy.gui_qt`) and the
dependency-free browser GUI (:mod:`pdbenergy.gui`) - drive the pipeline the same
way: they choose *arguments* and run the existing CLI in a child process.  All
decisions about what those arguments mean stay in the CLI, which is what makes
the command line shown in the log sufficient to reproduce a run by hand.

Keeping the mapping here means the two interfaces cannot drift apart, and that it
can be tested without starting a server or a window.

Raises :class:`ActionError` for unknown actions or missing required input; the
callers turn that into an HTTP 400 or a message box.
"""

from __future__ import annotations

import os
from typing import Any

#: Actions that need a non-empty list of PDB ids, a run directory, etc.
REQUIRED_INPUT: dict[str, tuple[str, ...]] = {
    "label": ("ids",),
    "evaluate": ("run_dir",),
    "predict": ("paths", "checkpoint"),
}

#: Which module each action runs.  Everything goes through the CLI package
#: except a couple of standalone tools under scripts/.
_SCRIPTS = {
    "measure_leakage": ("leakage measurement", "measure_leakage.py"),
    "estimate_workload": ("workload estimate", "estimate_workload.py"),
}


class ActionError(ValueError):
    """The request cannot be turned into a command (missing or unknown input)."""


def available_actions() -> list[str]:
    """Every action :func:`build_job` understands."""
    return [
        "download", "scan", "label", "dataset", "train", "evaluate", "predict",
        "ablate", "iterate", "measure_leakage", "estimate_workload",
    ]


def _require(payload: dict[str, Any], key: str, action: str):
    value = payload.get(key)
    if not value:
        raise ActionError(f"{action}: 缺少必需参数 {key!r}")
    return value


def build_job(
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    outputs_dir: str = "outputs",
    script_dir: str = "scripts",
) -> tuple[str, str, list[str]]:
    """Return ``(label, module, args)`` for one UI action.

    ``module`` is either a package module (run with ``python -m``) or a path to a
    ``.py`` file, which :class:`pdbenergy.gui.JobManager` distinguishes by suffix.
    """
    body = dict(payload or {})
    if action not in available_actions():
        raise ActionError(f"未知操作：{action!r}")

    if action in _SCRIPTS:
        label, filename = _SCRIPTS[action]
        return label, os.path.join(script_dir, filename), []

    if action == "download":
        args = ["download"]
        ids = [str(i) for i in (body.get("ids") or []) if str(i).strip()]
        if ids:
            args += ["--ids", *ids]
        if body.get("force"):
            args.append("--force")
        return "下载 PDB 条目", "pdbenergy.cli", args

    if action == "scan":
        args = ["inventory", "--no-table",
                "--max-residues", str(int(body.get("max_residues", 120))),
                "--min-residues", str(int(body.get("min_residues", 10)))]
        return "扫描 data/raw", "pdbenergy.cli", args

    if action == "label":
        ids = [str(i) for i in _require(body, "ids", action)]
        args = ["ensemble", "--ids", *ids,
                "--threads", str(int(body.get("threads", 8))),
                "--workers", str(int(body.get("workers", 1)))]
        if body.get("overwrite"):
            args.append("--overwrite")
        return f"标注 {len(ids)} 个条目", "pdbenergy.cli", args

    if action == "dataset":
        return "构建数据集切分", "pdbenergy.cli", ["dataset"]

    if action == "train":
        model = str(body.get("model", "schnet"))
        tag = str(body.get("tag") or f"{model}_protein")
        args = ["train", "--model", model, "--tag", tag]
        for flag, key, cast in (("--epochs", "epochs", int),
                                ("--batch-size", "batch_size", int),
                                ("--learning-rate", "learning_rate", float),
                                ("--hidden-dim", "hidden_dim", int),
                                ("--interactions", "interactions", int),
                                ("--split-mode", "split_mode", str)):
            value = body.get(key)
            # 0 / 0.0 / "" all mean "leave it at the config default", which the
            # CLI expresses by omitting the flag entirely.
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
        return f"训练 {model} → {tag}", "pdbenergy.cli", args

    if action == "evaluate":
        run_dir = str(_require(body, "run_dir", action))
        return (f"评估 {os.path.basename(run_dir.rstrip(os.sep))}",
                "pdbenergy.cli", ["evaluate", "--run-dir", run_dir])

    if action == "predict":
        paths = [str(p) for p in _require(body, "paths", action)]
        checkpoint = str(_require(body, "checkpoint", action))
        args = ["predict", *paths, "--checkpoint", checkpoint,
                "--max-models", str(int(body.get("max_models", 1))),
                "--threads", str(int(body.get("threads", 4))),
                "--json", os.path.join(outputs_dir, "predictions.json"),
                "--quiet"]
        if body.get("verify"):
            args.append("--verify")
        return f"预测 {len(paths)} 个文件", "pdbenergy.cli", args

    if action == "ablate":
        args = ["ablate", "--epochs", str(int(body.get("epochs", 20))),
                "--model", str(body.get("model", "mlp"))]
        return "数据泄漏消融", "pdbenergy.cli", args

    if action == "iterate":
        rounds = int(body.get("rounds", 1))
        args = ["--rounds", str(rounds)]
        args.append("--build-new" if body.get("build_new", True) else "--no-build-new")
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
        return f"迭代训练 ×{rounds}", "pdbenergy.iterate", args

    raise ActionError(f"未实现的操作：{action!r}")           # pragma: no cover
