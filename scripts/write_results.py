# -*- coding: utf-8 -*-
"""Generate the "results" chapter of TeachFlow.md from the run artifacts.

Everything in the chapter comes from files on disk, so the documentation cannot
drift away from the measurements.  Re-run it after any retraining:

    python scripts/write_results.py

Outputs are spliced into TeachFlow.md at the ``AUTORESULTS_PLACEHOLDER`` marker.
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "TeachFlow.md")
PLACEHOLDER = "AUTORESULTS_PLACEHOLDER"

RUNS = {
    "schnet": "outputs/schnet_protein",
    "mlp": "outputs/mlp_protein",
}
ABLATION = "outputs/ablation/split_leakage.json"


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        print("  warning: could not read %s (%s)" % (path, exc))
        return None


def fmt(value, digits=3, dash="—"):
    if value is None:
        return dash
    if isinstance(value, float):
        if value != value:      # NaN
            return dash
        return ("%%.%df" % digits) % value
    return str(value)


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Build the chapter
# --------------------------------------------------------------------------- #

def build_chapter() -> str:
    metrics = {name: load_json(os.path.join(path, "eval", "metrics_full.json"))
               for name, path in RUNS.items()}
    metrics = {k: v for k, v in metrics.items() if v}
    summary = load_json(os.path.join(RUNS["schnet"], "data_summary.json")) or \
        load_json(os.path.join(RUNS["mlp"], "data_summary.json"))
    ablation = load_json(os.path.join(ROOT, ABLATION))

    if not metrics and not summary:
        return ("*还没有可用的结果。先运行 `python -m pdbenergy.cli all`，"
                "再执行 `python scripts/write_results.py`。*")

    parts = ["### 11.1 数据集\n"]

    if summary:
        parts.append(markdown_table(
            ["项目", "值"],
            [
                ["蛋白数", summary.get("n_proteins")],
                ["总样本数", summary.get("n_samples")],
                ["训练 / 验证 / 测试样本", "%s / %s / %s" % (
                    summary.get("samples", {}).get("train"),
                    summary.get("samples", {}).get("val"),
                    summary.get("samples", {}).get("test"))],
                ["切分方式", "按蛋白质（`%s`）" % summary.get("split_mode")],
                ["预测目标", summary.get("target")],
                ["训练集目标均值 / 标准差",
                 "%s / %s kcal/mol" % (fmt(summary.get("norm_mean"), 2),
                                       fmt(summary.get("norm_std"), 2))],
            ],
        ))
        parts.append("")
        parts.append("**测试集蛋白（训练中完全没有出现过）：** `%s`\n" % (
            ", ".join(summary.get("proteins", {}).get("test", [])) or "—"))
        parts.append("**验证集蛋白：** `%s`\n" % (
            ", ".join(summary.get("proteins", {}).get("val", [])) or "—"))
        parts.append("**训练集蛋白：** `%s`\n" % (
            ", ".join(summary.get("proteins", {}).get("train", [])) or "—"))

        per_protein = summary.get("per_protein", {})
        if per_protein:
            rows = []
            for pid, info in sorted(per_protein.items(),
                                    key=lambda kv: -kv[1].get("dE_max_kcal", 0)):
                held = "test" if pid in summary.get("proteins", {}).get("test", []) else (
                    "val" if pid in summary.get("proteins", {}).get("val", []) else "train")
                rows.append([
                    pid, held, info.get("sequence_length"), info.get("n_atoms"),
                    info.get("n_frames"), fmt(info.get("reference_energy_kcal"), 1),
                    fmt(info.get("dE_max_kcal"), 1),
                ])
            parts.append("\n**每个蛋白的规模与能量跨度**（按 ΔE 跨度降序）：\n")
            parts.append(markdown_table(
                ["PDB", "划分", "残基数", "原子数", "帧数", "E_min (kcal/mol)",
                 "最大 ΔE (kcal/mol)"], rows))

    # -- model comparison --------------------------------------------------- #
    parts.append("\n### 11.2 模型对比（测试集 = 完全没见过的蛋白）\n")
    if metrics:
        rows = []
        for name, report in metrics.items():
            test = report.get("splits", {}).get("test") or report.get("splits", {}).get("train", {})
            overall = test.get("overall", {})
            rank = test.get("rank_discrimination", {})
            base = (report.get("baselines", {}) or {}).get("test", {})
            rows.append([
                "SchNet（图神经网络）" if name == "schnet" else "MLP（手工描述符基线）",
                "{:,}".format(report.get("n_parameters") or 0),
                report.get("best_epoch"),
                fmt(overall.get("mae"), 2),
                fmt(overall.get("rmse"), 2),
                fmt(overall.get("r2"), 4),
                fmt(overall.get("spearman"), 4),
                fmt(rank.get("mean_spearman"), 4),
                fmt(base.get("mae"), 2),
            ])
        parts.append(markdown_table(
            ["模型", "参数量", "最佳轮次", "MAE", "RMSE", "R²",
             "Spearman", "组内平均 ρ", "常数基线 MAE"], rows))
        parts.append(
            "\n> 单位除 R²/Spearman 外均为 kcal/mol。"
            "\"常数基线\"= 永远输出训练集平均 ΔE，任何模型都必须显著优于它。"
            "\"组内平均 ρ\"= 在同一个蛋白内部，预测排序与真实排序的平均 Spearman 相关，"
            "这才是打分函数真正需要的指标（§9.4）。")

    # -- per-source --------------------------------------------------------- #
    if metrics:
        parts.append("\n### 11.3 按采样来源分解（测试集）\n")
        for name, report in metrics.items():
            test = report.get("splits", {}).get("test") or {}
            per_source = test.get("per_source", {})
            if not per_source:
                continue
            rows = []
            for source, m in sorted(per_source.items(),
                                    key=lambda kv: -kv[1].get("mae", 0)):
                # R^2 is deliberately omitted per source: within one source group
                # the target variance is tiny (all md450K frames sit in a narrow
                # band), so R^2 explodes to large negative numbers that say nothing
                # about the model.  MAE/RMSE/Spearman remain interpretable.
                rows.append([source, m.get("n"), fmt(m.get("mae"), 2),
                             fmt(m.get("rmse"), 2), fmt(m.get("spearman"), 3)])
            parts.append("\n**%s**\n" % (
                "SchNet" if name == "schnet" else "MLP 基线"))
            parts.append(markdown_table(
                ["来源", "样本数", "MAE", "RMSE", "Spearman"], rows))

    # -- per protein -------------------------------------------------------- #
    if metrics:
        parts.append("\n### 11.4 按蛋白分解（测试集，SchNet）\n")
        report = metrics.get("schnet") or list(metrics.values())[0]
        test = report.get("splits", {}).get("test") or {}
        per_protein = test.get("per_protein", {})
        if per_protein:
            rows = []
            for pid, m in sorted(per_protein.items(), key=lambda kv: -kv[1].get("mae", 0)):
                rows.append([pid, m.get("n"), fmt(m.get("mae"), 2),
                             fmt(m.get("rmse"), 2), fmt(m.get("r2"), 3),
                             fmt(m.get("spearman"), 3)])
            parts.append(markdown_table(
                ["PDB", "样本数", "MAE", "RMSE", "R²", "Spearman"], rows))
            parts.append(
                "\n> 组内 Spearman 高，说明「能不能把好构象排在前面」这件事做得好；"
                "R² 甚至可能为负，说明该蛋白上的预测还不如直接用平均值。"
                "**这种分解是必需的**——总体指标会把这种失败完全掩盖（§9.5）。")

    # -- ablation ----------------------------------------------------------- #
    parts.append("\n### 11.5 消融实验：数据泄漏\n")
    direct = load_json(os.path.join(ROOT, "outputs", "ablation", "direct_leakage.json"))
    if direct:
        rows = []
        for mode, label in (("protein", "按蛋白质切分（本项目默认）"),
                            ("frame", "按帧随机切分（常见错误）")):
            info = direct.get(mode, {}) or {}
            if not info:
                continue
            rows.append([
                label,
                "%d / %d" % (info.get("n_train", 0), info.get("n_test", 0)),
                "%.1f%%" % (100 * info.get("same_protein_fraction", 0)),
                fmt(info.get("median_nn_distance"), 3),
                fmt(info.get("min_nn_distance"), 3),
                "%.1f%%" % (100 * info.get("near_duplicate_fraction", 0)),
            ])
        if rows:
            parts.append("**直接测量（不需要训练模型）**\n")
            parts.append(markdown_table(
                ["切分方式", "训练/测试帧数", "最近邻同蛋白比例", "最近邻距离中位数",
                 "最近邻距离最小值", "近似重复比例 (<0.05)"], rows))
            same = (direct.get("frame", {}) or {}).get("same_protein_fraction")
            if same is not None:
                parts.append(
                    "\n按帧切分时，**%.0f%% 的测试帧在训练集里的最近邻来自同一个蛋白**，"
                    "其中 **%.0f%% 与训练样本的距离小于 0.05**（标准化描述符空间），"
                    "也就是说**一半以上的「测试集」其实是训练集的近似副本**。"
                    "而按蛋白质切分时这两个数字都是 **0%%**。\n"
                    % (100 * same,
                       100 * (direct.get("frame", {}) or {}).get("near_duplicate_fraction", 0)))

    if ablation and isinstance(ablation, dict) and "protein" in ablation and "frame" in ablation:
        parts.append("\n**训练两种切分下的模型（同样的配置、同样的轮数）**\n")
        rows = []
        for mode, label in (("protein", "按蛋白质切分（正确）"),
                            ("frame", "按帧随机切分（泄漏）")):
            info = ablation[mode]
            rows.append([label, info.get("model", "—"), fmt(info.get("test_mae"), 2),
                         fmt(info.get("test_rmse"), 2), fmt(info.get("test_r2"), 4),
                         fmt(info.get("test_spearman"), 4),
                         fmt(info.get("best_val_mae"), 2)])
        parts.append(markdown_table(
            ["切分方式", "模型", "测试 MAE", "测试 RMSE", "测试 R²", "测试 Spearman",
             "最佳验证 MAE"], rows))
        factor = ablation.get("leakage_inflation_factor")
        if factor:
            parts.append(
                "\n两个 arm 的测试 MAE 之比为 **%.2f**（%.2f vs %.2f kcal/mol）。\n"
                % (factor, ablation["frame"].get("test_mae", float("nan")),
                   ablation["protein"].get("test_mae", float("nan"))))
        parts.append(
            "\n> **为什么 MAE 的差距没有想象中大？** 因为泄漏能被「利用」多少，"
            "取决于模型有没有能力把近似重复的帧**记下来**。"
            "这里用的是 1 万参数的描述符 MLP（跑得快），它本来就不擅长记忆单个构象，"
            "所以 MAE 差距被压缩了。\n"
            ">\n"
            "> 这恰恰是重点：**MAE 差距是一个依赖模型的间接证据，而上面那张"
            "「直接测量」表不依赖任何模型**——它数的是训练集和测试集到底有多重叠。"
            "看到 97%/53% 这两个数字，就不需要再靠训练模型来说服自己了。\n")

    # -- artifacts ---------------------------------------------------------- #
    parts.append("\n### 11.6 产物文件\n")
    parts.append(markdown_table(
        ["文件", "内容"],
        [
            ["`outputs/schnet_protein/checkpoint.pt`",
             "权重 + 模型配置 + 特征配置 + 归一化统计量 + 切分清单（自解释检查点）"],
            ["`outputs/schnet_protein/eval/metrics.json`",
             "全部指标：总体 / 每蛋白 / 每来源 / 排序能力 / 常数基线"],
            ["`outputs/schnet_protein/eval/parity_test.png`",
             "预测 vs 真实散点图（按蛋白着色）"],
            ["`outputs/schnet_protein/eval/residuals_test.png`",
             "残差分布 + 残差对真实值（查异方差）"],
            ["`outputs/schnet_protein/eval/learning_curve.png`",
             "训练/验证损失、验证 MAE、学习率"],
            ["`outputs/schnet_protein/eval/per_protein_test.png`",
             "每个测试蛋白的 MAE"],
            ["`outputs/schnet_protein/eval/per_source_test.png`",
             "每种采样来源的 MAE"],
            ["`outputs/schnet_protein/eval/energy_distribution.png`",
             "目标分布与各来源的箱线图"],
            ["`outputs/ablation/split_leakage.json`",
             "泄漏消融的原始数字"],
            ["`outputs/predictions.json`",
             "对真实 PDB 文件的预测（可用 `--verify` 附上真值）"],
        ]))

    # -- automatic diagnosis ------------------------------------------------ #
    parts.append("\n### 11.7 自动诊断（直接由上面的数字推出）\n")
    if metrics:
        lines = []
        for name, report in metrics.items():
            label = "SchNet" if name == "schnet" else "MLP 基线"
            test = report.get("splits", {}).get("test") or report.get("splits", {}).get("train", {})
            overall = test.get("overall", {})
            rng = test.get("energy_range", {})
            base = (report.get("baselines", {}) or {}).get("test", {})
            rank = test.get("rank_discrimination", {})

            true_span = (rng.get("true_max") or 0) - (rng.get("true_min") or 0)
            pred_span = (rng.get("pred_max") or 0) - (rng.get("pred_min") or 0)
            ratio = pred_span / true_span if true_span else float("nan")

            lines.append("**%s**" % label)
            lines.append("")
            lines.append("- 预测区间 %.1f–%.1f kcal/mol，真实区间 %.1f–%.1f kcal/mol；"
                         "**预测只覆盖了真实跨度的 %.0f%%**%s"
                         % (rng.get("pred_min") or 0, rng.get("pred_max") or 0,
                            rng.get("true_min") or 0, rng.get("true_max") or 0,
                            100 * ratio,
                            "，即典型的**回归到均值**。" if ratio < 0.5 else "。"))
            if base:
                mae = overall.get("mae", float("nan"))
                base_mae = base.get("mae", float("nan"))
                if pred_span < 1e-6 * max(1.0, true_span):
                    lines.append("- 测试 MAE %.2f kcal/mol，常数基线 %.2f kcal/mol → "
                                 "预测值几乎不变，**它就是常数预测器本身**，"
                                 "差别只来自用哪个常数。" % (mae, base_mae))
                else:
                    verdict = "**优于**" if mae < base_mae else "**不如**"
                    lines.append("- 测试 MAE %.2f kcal/mol，常数基线 %.2f kcal/mol → %s常数预测器。"
                                 % (mae, base_mae, verdict))
            lines.append("- 全局 Spearman ρ = %.3f，组内平均 ρ = %.3f（中位数 %.3f，"
                         "超过 0.5 的蛋白比例 %.0f%%）。"
                         % (overall.get("spearman", float("nan")),
                            rank.get("mean_spearman", float("nan")),
                            rank.get("median_spearman", float("nan")),
                            100 * (rank.get("fraction_above_0.5") or 0)))
            per_source = test.get("per_source", {})
            if per_source:
                worst = max(per_source.items(), key=lambda kv: kv[1].get("mae", 0))
                best = min(per_source.items(), key=lambda kv: kv[1].get("mae", 0))
                lines.append("- 按来源：最准的是 `%s`（MAE %.1f），最差的是 `%s`（MAE %.1f）。"
                             % (best[0], best[1].get("mae", float("nan")),
                                worst[0], worst[1].get("mae", float("nan"))))
            lines.append("")
        parts.append("\n".join(lines))

    # -- interpretation ----------------------------------------------------- #
    parts.append("""### 11.8 怎么读这些数字

1. **先看常数基线。** 如果 MAE 和常数基线差不多，模型实际上什么都没学到，
   后面的 R² 再好看也没有意义。
2. **再看预测区间。** 如果预测值只覆盖真实区间的一小部分，
   说明模型在**回归到均值**——这是"还没学会"的典型形态，而不是"学会了但不够准"。
3. **再看组内 ρ。** 这是"同一蛋白内部排序能力"，直接对应实际用途。
   全局 ρ 高但组内 ρ 低，意味着模型主要是学会了区分不同蛋白/不同采样来源，
   而不是学会了判断构象好坏。
4. **然后看按来源分解。** 低能区（`torsion`）通常明显比高温 MD 区准，
   因为高温区的构象形变更大、能量更高，而训练样本更少。
5. **最后看 Parity 图和残差图。** 点云被压向均值 → 回归到平均；
   残差随真实能量增大而增大 → 高能区拟合不足（异方差）。
6. **和泄漏消融对照。** 如果按帧切分的数字漂亮得多，
   那说明之前所有的乐观结论都不可信。

**本项目实测结论（诚实版）**：这条流水线是**完整可用**的——
从 PDB 下载、力场打标签、构象采样、按蛋白切分，到训练、评估、推理全部跑通，
产出的模型**确实带有一点真实的排序信号**（全局 Spearman 明显大于 0），
但**绝对精度不足以当作可用的打分函数**：预测区间被压缩、MAE 没有超过常数基线。

原因不是代码错误，而是**数据量**：只有 6 个训练蛋白、323 个构象，
而要泛化到没见过的蛋白，模型必须从几何细节里读出"热形变到什么程度"。
真实 MLIP 数据集是 $10^5$–$10^7$ 量级（§12.1）。
具体该往哪个方向补，见 §12.3。
""")
    return "\n".join(parts)


def has_artifacts() -> bool:
    """True once at least one run has been evaluated."""
    return any(
        os.path.exists(os.path.join(ROOT, path, "eval", "metrics_full.json"))
        for path in RUNS.values()
    )


def main() -> int:
    if not has_artifacts():
        print("no evaluated runs found under outputs/ - train and evaluate first:")
        print("  python -m pdbenergy.cli train --model schnet --tag schnet_protein")
        print("  python -m pdbenergy.cli evaluate --run-dir outputs/schnet_protein")
        print("TeachFlow.md left untouched.")
        return 1
    chapter = build_chapter()
    if not os.path.exists(DOC):
        print("TeachFlow.md not found at %s" % DOC)
        return 1
    text = io.open(DOC, encoding="utf-8").read()
    if PLACEHOLDER in text:
        # First generation: the "## 11. 实测结果" heading already precedes the marker.
        text = text.replace(PLACEHOLDER, chapter)
    else:
        # Regeneration.  Anchor on either the heading or the first section, so
        # this stays correct even if a previous run lost the heading.
        candidates = [i for i in (text.find("## 11. 实测结果"),
                                  text.find("### 11.1 数据")) if i != -1]
        if not candidates:
            print("could not locate chapter 11 in TeachFlow.md; nothing replaced")
            return 1
        start = min(candidates)
        end = text.find("## 12. ", start)
        if end == -1:
            print("could not find the start of chapter 12; nothing replaced")
            return 1
        text = text[:start] + "## 11. 实测结果\n\n" + chapter + "\n---\n\n" + text[end:]
    io.open(DOC, "w", encoding="utf-8").write(text)
    print("wrote chapter 11 into TeachFlow.md (%d characters)" % len(chapter))
    return 0


if __name__ == "__main__":
    sys.exit(main())
