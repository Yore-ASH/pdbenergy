"""PySide6 desktop GUI: 训练 / 预测 / 测试。

    pdbenergy gui            # 或 python -m pdbenergy.gui_qt
    pdbenergy-gui

与浏览器版（``pdbenergy web``）的关系
------------------------------------
两者共用同一套底层：

* :mod:`pdbenergy.actions` 把界面动作翻译成命令行 —— 唯一的真相来源，
  所以两个界面不会各自漂移；
* :class:`pdbenergy.gui.JobManager` 负责起子进程、按行回传输出、真正取消；
* :mod:`pdbenergy.gui` 里的只读自省函数（项目状态、运行列表、检查点列表）也是共享的。

界面只负责"选参数"，逻辑全在 CLI 里。因此日志里显示的命令行可以直接粘到终端重跑。

字体约定
--------
正文用 Times New Roman（含中文回退）；**代码区一律等宽**：实时日志、命令行预览、
以及路径类输入框。日志用等宽是刚需——力场输出是数字列，比例字体会让它们参差不齐。

设计要点
--------
* 长任务跑在子进程里，用 ``QTimer`` 轮询输出。界面线程从不阻塞，
  取消是真的 ``terminate``；关闭窗口时会终止所有还在跑的子任务。
* 每个需要做决定的控件都有 ``setToolTip`` 注解（Web 版是悬停气泡，这里用原生提示）。
* 帮助页用 ``QTextBrowser`` 承载常见问题与术语表。
"""

from __future__ import annotations

# --- 允许直接运行本文件：IDE 的 Run 按钮 / 双击 / python pdbenergy\gui_qt.py --- #
# 相对导入需要一个父包。`python -m pdbenergy.gui_qt` 有父包，直接跑文件没有，
# 于是停在第一条 `from .x import y` 上并报：
#     ImportError: attempted relative import with no known parent package
# 所以这种情况先把项目根目录放进 sys.path，再把自己当作包模块重新派发一次。
# 这一段必须在任何相对导入之前。
if __package__ in (None, ""):                                    # pragma: no cover
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pdbenergy.gui_qt import main as _main

    _sys.exit(_main())

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------- #
# Qt imports (kept together so the missing-dependency error is easy to produce)
# --------------------------------------------------------------------------- #

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QAction, QFont, QFontDatabase, QKeySequence, QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:                                   # pragma: no cover
    raise SystemExit(
        "PySide6 未安装。安装方式：\n"
        '    pip install -e ".[gui]"\n'
        "    pip install -r requirements-gui.txt\n"
        "不装也可以使用命令行，或不用 Qt 的浏览器版界面：pdbenergy web\n"
        f"(原始错误：{exc})"
    ) from exc

from .actions import ActionError, build_job
from .config import Config
from .gui import (                       # 与浏览器版共享的只读自省
    JobManager,
    list_pdb_files,
    list_runs,
    predict_checkpoints,
    project_state,
    resolve_dir,
)

# --------------------------------------------------------------------------- #
# 小工具：控件工厂（让页面代码短且一致）
# --------------------------------------------------------------------------- #

MONO_FAMILIES = ("Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Courier New")


def monospace_font(size: int = 10) -> QFont:
    """等宽字体，用于日志与代码类区域。"""
    font = QFont()
    for family in MONO_FAMILIES:
        if family in QFontDatabase.families():
            font.setFamily(family)
            break
    else:                                   # 谁都没有就用系统固定宽度字体
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSize(size)
    font.setStyleHint(QFont.Monospace)
    return font


def label(text: str, tip: str | None = None) -> QLabel:
    w = QLabel(text)
    if tip:
        w.setToolTip(tip)
    return w


def spin(value: int, lo: int, hi: int, *, tip: str | None = None, step: int = 1) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setSingleStep(step)
    if tip:
        w.setToolTip(tip)
    return w


def dspin(value: float, lo: float, hi: float, *, decimals: int = 4,
          step: float = 1e-4, tip: str | None = None) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setValue(value)
    w.setSingleStep(step)
    if tip:
        w.setToolTip(tip)
    return w


def combo(items: Sequence[tuple[str, Any]], *, tip: str | None = None) -> QComboBox:
    w = QComboBox()
    for text, data in items:
        w.addItem(text, data)
    if tip:
        w.setToolTip(tip)
    return w


def check(text: str, checked: bool = False, *, tip: str | None = None) -> QCheckBox:
    w = QCheckBox(text)
    w.setChecked(checked)
    if tip:
        w.setToolTip(tip)
    return w


def button(text: str, on_click: Callable[[], None], *, tip: str | None = None,
           primary: bool = False) -> QPushButton:
    w = QPushButton(text)
    w.clicked.connect(on_click)
    if tip:
        w.setToolTip(tip)
    if primary:
        w.setDefault(True)
    return w


def mono_line(placeholder: str = "", *, tip: str | None = None) -> QLineEdit:
    """路径类输入框，用等宽字体（内容是路径/文件，属于"代码区"）。"""
    w = QLineEdit()
    w.setPlaceholderText(placeholder)
    w.setFont(monospace_font(9))
    if tip:
        w.setToolTip(tip)
    return w


def form(rows: Sequence[tuple[str, QWidget, str | None]]) -> QWidget:
    """(标签, 控件, 注解) 三元的列表 -> 一个 QFormLayout 容器。"""
    box = QWidget()
    layout = QFormLayout(box)
    layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for text, widget, tip in rows:
        layout.addRow(label(text, tip), widget)
    return box


def grid(rows: Sequence[Sequence[tuple[str, QWidget, str | None]]]) -> QWidget:
    """二维网格；每个格子是 (标签, 控件, 注解)，用于并排的设置项。"""
    box = QWidget()
    layout = QGridLayout(box)
    for r, row in enumerate(rows):
        for c, (text, widget, tip) in enumerate(row):
            cell = QWidget()
            inner = QVBoxLayout(cell)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(2)
            inner.addWidget(label(text, tip))
            inner.addWidget(widget)
            layout.addWidget(cell, r, c)
    return box


def table(headers: Sequence[str]) -> QTableWidget:
    w = QTableWidget(0, len(headers))
    w.setHorizontalHeaderLabels(list(headers))
    w.verticalHeader().setVisible(False)
    w.setEditTriggers(QAbstractItemView.NoEditTriggers)
    w.setSelectionBehavior(QAbstractItemView.SelectRows)
    w.setAlternatingRowColors(True)
    w.horizontalHeader().setStretchLastSection(True)
    return w


def fill_table(widget: QTableWidget, rows: Sequence[Sequence[Any]]) -> None:
    widget.setRowCount(0)
    for row in rows:
        r = widget.rowCount()
        widget.insertRow(r)
        for c, value in enumerate(row):
            item = QTableWidgetItem("—" if value is None else str(value))
            if c == 0:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            else:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            widget.setItem(r, c, item)
    widget.resizeColumnsToContents()


def num(value: Any, digits: int = 2) -> str:
    """None / NaN 一律显示为破折号 —— 指标未定义时不该显示 'nan'。"""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:                                   # NaN
        return "—"
    return f"{f:.{digits}f}"


# --------------------------------------------------------------------------- #
# 共享上下文
# --------------------------------------------------------------------------- #


@dataclass
class AppContext:
    """界面各页面共享的东西。"""

    cfg: Config
    dirs: dict[str, str]
    manager: JobManager
    window: "MainWindow | None" = None

    #: 最近一次 project_state() 的结果
    state: dict[str, Any] | None = None

    def refresh(self) -> dict[str, Any]:
        self.state = project_state(
            self.cfg,
            interim_dir=self.dirs["interim"],
            raw_dir=self.dirs["raw"],
            outputs_dir=self.dirs["outputs"],
            processed_dir=self.dirs["processed"],
        )
        return self.state

    def run(self, action: str, payload: dict[str, Any] | None = None) -> None:
        """把动作翻译成命令行并启动；错误用对话框显示，不静默失败。"""
        try:
            label_text, module, args = build_job(
                action, payload, outputs_dir=self.dirs["outputs"]
            )
        except ActionError as exc:
            QMessageBox.warning(self.window, "无法启动", str(exc))
            return
        assert self.window is not None
        self.window.start_job(label_text, module, args)


# --------------------------------------------------------------------------- #
# 页面基类
# --------------------------------------------------------------------------- #


class Page(QWidget):
    """每个页面继承它；``update_state`` 在项目状态刷新时被调用。"""

    title = "页面"

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(16, 14, 16, 14)
        self.root.setSpacing(10)

    def heading(self, text: str, hint: str = "") -> None:
        h = QLabel(f"<b style='font-size:15px'>{text}</b>")
        self.root.addWidget(h)
        if hint:
            note = QLabel(hint)
            note.setWordWrap(True)
            note.setStyleSheet("color:#666;")
            self.root.addWidget(note)

    def update_state(self, state: dict[str, Any]) -> None:      # noqa: B027
        """默认什么都不做。"""

    def on_job_finished(self, status: str) -> None:             # noqa: B027
        """任务结束时被调用（成功/失败/取消都会调用）。

        实现里**绝不能弹模态对话框**：这是自动回调，没有用户去点"确定"。
        早期版本在这里弹出"还没有体检报告"，结果任何任务一结束界面就卡死
        （无人值守运行时尤其明显）。要提示就调 :meth:`notify` 写日志。
        """

    def notify(self, text: str) -> None:
        """非模态提示：写进底部日志。"""
        if self.ctx.window is not None:
            self.ctx.window.log_message(f"[{self.title}] {text}")


# --------------------------------------------------------------------------- #
# 概览
# --------------------------------------------------------------------------- #


class OverviewPage(Page):
    title = "概览"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.heading("概览", "项目当前状态。所有页面共享底部同一个实时日志。")

        box = QGroupBox("项目")
        self.info = QFormLayout(box)
        self.info.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.root.addWidget(box)

        self.runs = table(["运行", "切分/目标", "蛋白", "构象", "验证 MAE",
                           "测试 MAE", "测试 ρ", "状态"])
        self.runs.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.root.addWidget(QLabel("<b>已有运行</b>"))
        self.root.addWidget(self.runs, 1)

    def update_state(self, state: dict[str, Any]) -> None:
        while self.info.rowCount():
            self.info.removeRow(0)
        counts = state.get("counts", {})
        inv = state.get("inventory")
        rows = [
            ("工作目录", state.get("cwd", "")),
            ("data/raw 里的 PDB 文件", str(counts.get("raw_pdb", 0))),
            ("已打标签（data/interim）", str(counts.get("labelled", 0))),
            ("待标注", str(counts.get("pending", 0))),
            ("体检结果", (f'扫描 {inv["scanned"]} 条，可用 {inv["usable"]}，'
                          f'排除 {inv["unusable"]}') if inv else "未扫描"),
            ("可用条目的残基范围",
             f'{inv["min_residues"]} – {inv["max_residues"]}' if inv else "—"),
        ]
        for key, value in rows:
            field = QLabel(value)
            field.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.info.addRow(label(key), field)

        data = []
        for run in state.get("runs", []):
            metrics = run.get("metrics") or {}
            val = metrics.get("val") or {}
            test = metrics.get("test") or {}
            summary = run.get("summary") or {}
            data.append([
                run["name"],
                f'{summary.get("split_mode", "?")}/{summary.get("target", "?")}',
                run.get("n_proteins"), run.get("n_samples"),
                num(val.get("mae")), num(test.get("mae")),
                num(test.get("spearman"), 3),
                "已评估" if run.get("has_eval") else "未评估",
            ])
        fill_table(self.runs, data)


# --------------------------------------------------------------------------- #
# 数据 / 打标签
# --------------------------------------------------------------------------- #


class DataPage(Page):
    title = "数据 / 打标签"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        defaults = (ctx.state or ctx.refresh())["defaults"]
        self.heading(
            "数据 / 打标签",
            "扫描会解析 data/raw 里每个文件（几千个文件要几分钟），过滤掉非蛋白、"
            "过短和过长的条目。打标签是最慢的一步，实测每个蛋白 2–4 分钟；"
            "已完成的会跳过，可以分批做。",
        )

        self.max_res = spin(defaults["prepare"]["max_residues"], 5, 10000, step=10,
                            tip="只处理不超过这个残基数的条目。大蛋白的构建代价超线性增长"
                                "（几千残基的条目会让 OpenMM 几乎跑不完）")
        self.min_res = spin(defaults["prepare"]["min_residues"], 0, 500,
                            tip="过滤掉单残基、二肽这类没有构象能可学的小碎片")
        self.threads = spin(8, 1, 64, tip="OpenMM 内部使用的并行线程数。实测 8 线程比 1 线程"
                                         "快约 3.5–4 倍")
        self.workers = spin(1, 1, 32, tip="同时处理多少个蛋白（多进程）。"
                                          "workers × threads 不要超过物理核数，否则互相抢核更慢")

        controls = QGroupBox("扫描与标注设置")
        lay = QVBoxLayout(controls)
        lay.addWidget(grid([
            [("最大残基数 max_residues", self.max_res, None),
             ("最小残基数 min_residues", self.min_res, None),
             ("OpenMM 线程数", self.threads, None),
             ("并行进程数 workers", self.workers, None)],
        ]))
        buttons = QHBoxLayout()
        buttons.addWidget(button("扫描 data/raw", self.do_scan, primary=True,
                                 tip="解析 data/raw 全部文件并生成体检报告，"
                                     "结果决定后续能标注哪些条目"))
        buttons.addWidget(button("列出待标注", self.load_pending,
                                 tip="按当前残基上下限，列出还没标注、且值得标注的条目"))
        buttons.addWidget(button("下载内置条目", lambda: self.ctx.run("download"),
                                 tip="从 RCSB 下载内置的那批小蛋白条目（需要联网）"))
        buttons.addWidget(button("构建数据集切分", lambda: self.ctx.run("dataset"),
                                 tip="按蛋白质切分训练/验证/测试集，并计算目标的均值与标准差"))
        buttons.addWidget(button("工作量估算", lambda: self.ctx.run("estimate_workload"),
                                 tip="推算不同残基上限下要标多少条、多少构象、多少磁盘、多长时间"))
        buttons.addStretch(1)
        lay.addLayout(buttons)
        self.root.addWidget(controls)

        self.count = QLabel("未加载")
        self.root.addWidget(self.count)

        self.entries = table(["选择", "PDB 编号", "残基数", "原子数", "模型数",
                              "实验方法", "备注"])
        self.entries.setSelectionMode(QAbstractItemView.NoSelection)
        self.root.addWidget(self.entries, 1)

        act = QHBoxLayout()
        act.addWidget(button("全选", lambda: self._select_all(True)))
        act.addWidget(button("清空", lambda: self._select_all(False)))
        act.addWidget(label("本次最多标注",
                            "本次最多标注多少条。剩余的下次继续，已完成的会自动跳过"))
        self.limit = spin(50, 0, 100000, step=10)
        act.addWidget(self.limit)
        act.addWidget(button("标注已勾选", self.do_label, primary=True,
                             tip="对勾选的条目启动力场打标签（会跑 OpenMM，耗时最长）"))
        act.addStretch(1)
        self.root.addLayout(act)

    # -- 动作 --------------------------------------------------------------- #
    def do_scan(self) -> None:
        self.ctx.run("scan", {"max_residues": self.max_res.value(),
                              "min_residues": self.min_res.value()})

    def load_pending(self, silent: bool = False) -> None:
        """列出待标注条目。``silent=True`` 时只用日志提示，不弹窗（自动回调路径）。"""
        inventory = os.path.join(self.ctx.dirs["outputs"], "inventory.json")
        if not os.path.exists(inventory):
            if silent:
                self.notify("还没有体检报告；点「扫描 data/raw」后会列出待标注条目。")
            else:
                QMessageBox.information(self, "还没有体检报告",
                                        "请先点「扫描 data/raw」。")
            return
        import json

        from .gui import _usable_at

        with open(inventory, "r", encoding="utf-8") as fh:
            reports = json.load(fh)
        labelled = {
            name[:-4] for name in os.listdir(self.ctx.dirs["interim"])
            if name.endswith(".npz")
        } if os.path.isdir(self.ctx.dirs["interim"]) else set()
        rows = [
            r for r in reports
            if _usable_at(r, self.max_res.value(), self.min_res.value())
            and r["pdb_id"] not in labelled
        ]
        rows.sort(key=lambda r: r.get("n_residues", 0))
        self._show_entries(rows)

    def _show_entries(self, rows: list[dict[str, Any]]) -> None:
        self.count.setText(f"待标注 {len(rows)} 条（勾选后点「标注已勾选」）")
        self.entries.setRowCount(0)
        for row in rows:
            r = self.entries.rowCount()
            self.entries.insertRow(r)
            box = QCheckBox()
            holder = QWidget()
            lay = QHBoxLayout(holder)
            lay.setContentsMargins(6, 0, 0, 0)
            lay.addWidget(box)
            self.entries.setCellWidget(r, 0, holder)
            values = [row.get("pdb_id", ""), row.get("n_residues"),
                      row.get("n_atoms"), row.get("n_models"),
                      (row.get("experiment") or "")[:24], row.get("reason") or ""]
            for c, value in enumerate(values, start=1):
                self.entries.setItem(r, c, QTableWidgetItem(str(value)))
        self.entries.resizeColumnsToContents()

    def _select_all(self, on: bool) -> None:
        for r in range(self.entries.rowCount()):
            holder = self.entries.cellWidget(r, 0)
            if holder:
                box = holder.findChild(QCheckBox)
                if box:
                    box.setChecked(on)

    def _checked_ids(self) -> list[str]:
        ids = []
        for r in range(self.entries.rowCount()):
            holder = self.entries.cellWidget(r, 0)
            box = holder.findChild(QCheckBox) if holder else None
            item = self.entries.item(r, 1)
            if box and box.isChecked() and item is not None:
                ids.append(item.text())
        return ids

    def do_label(self) -> None:
        ids = self._checked_ids()
        if not ids:
            QMessageBox.information(self, "没有勾选", "请先勾选要标注的条目（可点「全选」）。")
            return
        limit = self.limit.value()
        chosen = ids[:limit] if limit > 0 else ids
        if limit > 0 and len(ids) > limit:
            self.ctx.window.log_message(
                f"本次只标注前 {limit} 条（共勾选 {len(ids)} 条，可在输入框调整）"
            )
        self.ctx.run("label", {"ids": chosen, "threads": self.threads.value(),
                               "workers": self.workers.value()})

    def on_job_finished(self, status: str) -> None:
        if status == "done":
            self.load_pending(silent=True)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #


class TrainPage(Page):
    title = "训练"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        defaults = (ctx.state or ctx.refresh())["defaults"]
        model, train = defaults["model"], defaults["train"]
        self.heading(
            "训练",
            "超参改动会写进命令行并显示在日志里，所以每次运行都可手工复现。"
            "改 hidden_dim / cutoff / n_rbf 会使图缓存失效并改变权重形状——"
            "那种情况下不能热启动。",
        )

        self.model = combo([("SchNet（图神经网络）", "schnet"),
                            ("MLP（手工描述符基线）", "mlp")],
                           tip="SchNet：消息传递图神经网络，直接吃 3D 结构。"
                               "MLP：用 Rg、接触数等 24 维全局量的基线")
        self.epochs = spin(train["epochs"], 1, 100000,
                           tip="本次把训练集完整过几遍。配置里的 epochs 永远是"
                               "「这次再跑多少轮」，续训时不会重复已跑的轮次")
        self.batch = spin(train["batch_size"], 1, 4096,
                          tip="一次梯度更新用多少个构象。大 batch 的矩阵运算效率更高"
                              "（实测每样本耗时从 2.13 s 降到 1.36 s），但更占内存")
        self.lr = dspin(train["learning_rate"], 1e-6, 1.0,
                        tip="AdamW 的步长。太大不收敛，太小训练慢")
        self.hidden = spin(model["hidden_dim"], 4, 4096, step=8,
                           tip="每个原子的特征向量维度。越大容量越强、越慢，"
                               "也越容易在小数据集上过拟合")
        self.inter = spin(model["n_interactions"], 1, 16,
                          tip="消息传递层数。一个原子能「看」到约 层数 × cutoff 的范围"
                              "（本项目 3 × 4 Å = 12 Å）")
        self.split = combo([("按蛋白质（正确）", "protein"),
                            ("按帧（数据泄漏，仅用于对照）", "frame")],
                           tip="按蛋白质切分：测试集蛋白训练中从未出现，是唯一诚实的做法。"
                               "按帧切分：同一蛋白的近似重复构象会同时出现在训练和测试里，"
                               "指标会被严重高估")
        self.tag = QLineEdit("schnet_protein")
        self.tag.setFont(monospace_font(9))
        self.tag.setToolTip("输出目录名，结果写到 outputs/<tag>/")

        self.nocache = check("禁用图缓存", False,
                             tip="不把图缓存进内存，改成每次现场构造。"
                                 "数据集超过约 2 万构象时必须打开（缓存会 OOM）")
        self.recompute = check("重算归一化", False,
                               tip="热启动/续训时默认沿用检查点里的目标均值与标准差。"
                                   "如果新蛋白的分布确实不同，才需要重算")
        self.init_from = mono_line("outputs/.../checkpoint.pt",
                                   tip="热启动：用旧模型的权重初始化，"
                                       "但优化器、轮次计数、历史全部重来。数据变多时用")
        self.resume = mono_line("outputs/某个运行目录",
                                tip="续训：连 AdamW 的两个动量和轮次计数一起恢复，"
                                    "接着往下跑。长时间训练被中断后用它")
        pick_init = button("浏览…", lambda: self._pick(self.init_from))
        pick_resume = button("浏览…", lambda: self._pick(self.resume, directory=True))

        box = QGroupBox("超参数")
        lay = QVBoxLayout(box)
        lay.addWidget(grid([
            [("模型", self.model, None), ("训练轮数 epochs", self.epochs, None),
             ("批大小 batch_size", self.batch, None),
             ("学习率 learning_rate", self.lr, None)],
            [("隐藏维度 hidden_dim", self.hidden, None),
             ("交互层数 n_interactions", self.inter, None),
             ("数据切分方式", self.split, None),
             ("输出目录名 tag", self.tag, None)],
        ]))
        row = QHBoxLayout()
        row.addWidget(self.nocache)
        row.addWidget(self.recompute)
        row.addStretch(1)
        lay.addLayout(row)

        cont = QHBoxLayout()
        cont.addWidget(label("热启动 --init-from", self.init_from.toolTip()))
        cont.addWidget(self.init_from, 1)
        cont.addWidget(pick_init)
        lay.addLayout(cont)
        cont2 = QHBoxLayout()
        cont2.addWidget(label("续训 --resume", self.resume.toolTip()))
        cont2.addWidget(self.resume, 1)
        cont2.addWidget(pick_resume)
        lay.addLayout(cont2)
        lay.addWidget(label("两者只能填一个。留空 = 从头训练。"))

        self.root.addWidget(box)
        self.root.addWidget(button("开始训练", self.do_train, primary=True,
                                   tip="启动训练。任务在子进程里跑，界面不会卡，日志实时回传"))

        self.figs = QScrollArea()
        self.figs.setWidgetResizable(True)
        self.figs.setMinimumHeight(200)
        self.figs.setWidget(QLabel("训练完成后这里显示学习曲线"))
        self.root.addWidget(QLabel("<b>最近运行的学习曲线</b>"))
        self.root.addWidget(self.figs, 1)

    def _pick(self, target: QLineEdit, directory: bool = False) -> None:
        if directory:
            path = QFileDialog.getExistingDirectory(self, "选择运行目录",
                                                   self.ctx.dirs["outputs"])
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择检查点",
                                                  self.ctx.dirs["outputs"],
                                                  "PyTorch 检查点 (*.pt)")
        if path:
            target.setText(os.path.relpath(path, os.getcwd()).replace("\\", "/")
                           if path.startswith(os.getcwd()) else path)

    def do_train(self) -> None:
        init, resume = self.init_from.text().strip(), self.resume.text().strip()
        if init and resume:
            QMessageBox.warning(self, "参数冲突", "热启动和续训只能填一个。")
            return
        self.ctx.run("train", {
            "model": self.model.currentData(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch.value(),
            "learning_rate": self.lr.value(),
            "hidden_dim": self.hidden.value(),
            "interactions": self.inter.value(),
            "split_mode": self.split.currentData(),
            "tag": self.tag.text().strip(),
            "init_from": init or None,
            "resume": resume or None,
            "no_cache_graphs": self.nocache.isChecked(),
            "recompute_normalisation": self.recompute.isChecked(),
        })

    def update_state(self, state: dict[str, Any]) -> None:
        curves = []
        for run in state.get("runs", [])[-3:]:
            if "learning_curve" in (run.get("figures") or []):
                curves.append((run["name"],
                               os.path.join(run["path"], "eval", "learning_curve.png")))
        if not curves:
            return
        holder = QWidget()
        lay = QHBoxLayout(holder)
        for name, path in curves:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            cell = QVBoxLayout()
            pic = QLabel()
            pic.setPixmap(pixmap.scaledToWidth(360, Qt.SmoothTransformation))
            cell.addWidget(pic)
            caption = QLabel(name)
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet("color:#666;")
            cell.addWidget(caption)
            wrap = QWidget()
            wrap.setLayout(cell)
            lay.addWidget(wrap)
        lay.addStretch(1)
        self.figs.setWidget(holder)


# --------------------------------------------------------------------------- #
# 评估 / 测试
# --------------------------------------------------------------------------- #


class EvalPage(Page):
    title = "评估 / 测试"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.heading(
            "评估 / 测试",
            "选一个运行做评估并看图。注意：MAE 单独看会骗人——务必同时看"
            "「预测跨度」和「组内 ρ」。预测跨度只有百分之几，说明模型基本在输出常数。",
        )

        self.run = combo([], tip="outputs/ 下每个含 checkpoint.pt 的目录")
        self.epochs = spin(20, 1, 10000, tip="数据泄漏消融实验的训练轮数")
        box = QGroupBox("评估设置")
        lay = QVBoxLayout(box)
        lay.addWidget(grid([[
            ("运行", self.run, None), ("消融轮数", self.epochs, None),
        ]]))
        row = QHBoxLayout()
        row.addWidget(button("评估该运行", self.do_eval, primary=True,
                             tip="在验证集和测试集上算指标，并生成 parity、残差、"
                                 "学习曲线等图"))
        row.addWidget(button("数据泄漏消融", self.do_ablate,
                             tip="用同样配置只改切分方式训练两次，量化「按帧切分」"
                                 "能把指标虚高多少"))
        row.addStretch(1)
        lay.addLayout(row)
        self.root.addWidget(box)

        self.summary = QTextBrowser()
        self.summary.setMaximumHeight(190)
        self.summary.setOpenExternalLinks(True)
        self.root.addWidget(self.summary)

        self.figure_picker = combo([], tip="选择要查看的图")
        self.figure_picker.currentIndexChanged.connect(self.show_figure)
        pick_row = QHBoxLayout()
        pick_row.addWidget(label("图表"), 0)
        pick_row.addWidget(self.figure_picker, 1)
        self.root.addLayout(pick_row)

        self.image = QLabel("选择运行后这里显示图表")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(320)
        self.image.setStyleSheet("background:#fafafa;border:1px solid #ddd;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image)
        self.root.addWidget(scroll, 1)

    def _current(self) -> dict[str, Any] | None:
        name = self.run.currentData()
        for run in (self.ctx.state or {}).get("runs", []):
            if run["name"] == name:
                return run
        return None

    def do_eval(self) -> None:
        run = self._current()
        if not run:
            QMessageBox.information(self, "没有选择运行", "请先训练一个模型，或在下拉框里选择。")
            return
        self.ctx.run("evaluate", {"run_dir": run["path"]})

    def do_ablate(self) -> None:
        self.ctx.run("ablate", {"epochs": self.epochs.value(), "model": "mlp"})

    def update_state(self, state: dict[str, Any]) -> None:
        current = self.run.currentData()
        self.run.clear()
        for run in state.get("runs", []):
            suffix = " ✓" if run.get("has_eval") else ""
            self.run.addItem(run["name"] + suffix, run["name"])
        if current:
            index = self.run.findData(current)
            if index >= 0:
                self.run.setCurrentIndex(index)
        self.render()

    def render(self) -> None:
        run = self._current()
        if not run:
            self.summary.setHtml("<p>未选择运行</p>")
            self.figure_picker.clear()
            self.image.setText("选择运行后这里显示图表")
            return
        if not run.get("has_eval"):
            self.summary.setHtml("<p>该运行还没评估过，点上面的「评估该运行」。</p>")
            self.figure_picker.clear()
            self.image.setText("尚无图表")
            return

        metrics = run.get("metrics") or {}
        baselines = run.get("baselines") or {}
        html = [f"<p><b>参数量</b> {run.get('n_parameters', '—')}</p><table cellpadding=4>"]
        html.append("<tr><th align=left>数据集</th><th>MAE</th><th>RMSE</th>"
                    "<th>R²</th><th>Spearman</th><th>组内 ρ</th><th>常数基线 MAE</th></tr>")
        for split, cn in (("val", "验证"), ("test", "测试")):
            m = metrics.get(split)
            if not m:
                continue
            base = (baselines.get(split) or {}).get("mae")
            verdict = ""
            if base is not None and m.get("mae") is not None:
                verdict = ("<span style='color:#127a2b'>优于基线</span>"
                           if m["mae"] < base else
                           "<span style='color:#b3261e'>不如基线</span>")
            rank = m.get("rank_rho")
            rank_cell = num(rank, 4)
            if rank is not None and rank < 0.3:
                rank_cell += " <span style='color:#b26a00'>←排序能力弱</span>"
            html.append(
                f"<tr><td>{cn}</td><td>{num(m.get('mae'))}</td><td>{num(m.get('rmse'))}</td>"
                f"<td>{num(m.get('r2'), 4)}</td><td>{num(m.get('spearman'), 4)}</td>"
                f"<td>{rank_cell}</td><td>{num(base)} {verdict}</td></tr>"
            )
        html.append("</table>")
        html.append("<p style='color:#666'>预测跨度不在 metrics.json 里：看 parity 图，"
                    "或跑 scripts/compare_configs.py 来确认模型是否在输出常数。"
                    "详见「帮助」页。</p>")
        self.summary.setHtml("".join(html))

        figures = run.get("figures") or []
        previous = self.figure_picker.currentData()
        self.figure_picker.clear()
        for name in figures:
            self.figure_picker.addItem(name, name)
        if previous:
            index = self.figure_picker.findData(previous)
            if index >= 0:
                self.figure_picker.setCurrentIndex(index)
        self.show_figure()

    def show_figure(self) -> None:
        run, name = self._current(), self.figure_picker.currentData()
        if not run or not name:
            self.image.setText("选择图表后在这里显示")
            return
        path = os.path.join(run["path"], "eval", f"{name}.png")
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image.setText(f"读不到图片：{path}")
            return
        self.image.setPixmap(pixmap.scaledToWidth(
            max(320, self.image.width() - 20), Qt.SmoothTransformation))

    def on_job_finished(self, status: str) -> None:
        if status == "done":
            self.ctx.window.refresh_state()


# --------------------------------------------------------------------------- #
# 预测
# --------------------------------------------------------------------------- #


class PredictPage(Page):
    title = "预测"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.heading(
            "预测",
            "给 PDB 文件打分并排序。输出的是相对构象能 ΔE = E − min(E)，"
            "只在同一分子内部可比。勾选「对照真值」会额外用 OpenMM 算一次真值能量，"
            "可以直接看到误差（慢一些）。",
        )

        self.checkpoint = combo([], tip="模型检查点：权重 + 模型/特征配置 + 归一化统计量。"
                                        "单个文件就足以做推理")
        self.max_models = spin(1, 1, 200,
                               tip="NMR 条目一个文件里有几十个模型，这里限制每个文件取前几个")
        self.threads = spin(4, 1, 64, tip="对照真值时 OpenMM 使用的线程数")
        self.verify = check("对照真值（--verify）", False,
                            tip="额外用物理力场算一次真值能量并显示误差。"
                                "这是检验模型是否可用的唯一诚实方式")
        self.manual = mono_line("data/raw/1L2Y.pdb,data/raw/1CRN.pdb",
                                tip="也可以直接手填路径，多个用英文逗号分隔")

        box = QGroupBox("预测设置")
        lay = QVBoxLayout(box)
        lay.addWidget(grid([[
            ("检查点 checkpoint", self.checkpoint, None),
            ("每个文件取前 N 个模型", self.max_models, None),
            ("OpenMM 线程数", self.threads, None),
        ]]))
        row = QHBoxLayout()
        row.addWidget(self.verify)
        row.addWidget(button("开始预测", self.do_predict, primary=True,
                             tip="对选中的 PDB 文件逐个预测并按预测能量排序"))
        row.addWidget(button("读取上次结果", self.load_results))
        row.addStretch(1)
        lay.addLayout(row)
        manual_row = QHBoxLayout()
        manual_row.addWidget(label("或手填文件路径", self.manual.toolTip()))
        manual_row.addWidget(self.manual, 1)
        lay.addLayout(manual_row)
        self.root.addWidget(box)

        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.files.setMaximumHeight(150)
        self.root.addWidget(QLabel("<b>从 data/raw 选择文件（可多选）</b>"))
        self.root.addWidget(self.files)

        self.results = table(["文件", "模型序号", "预测 ΔE", "真值 ΔE", "误差", "残基数"])
        self.root.addWidget(QLabel("<b>预测结果（按预测能量排序）</b>"))
        self.root.addWidget(self.results, 1)

    def do_predict(self) -> None:
        paths = [item.text() for item in self.files.selectedItems()]
        manual = self.manual.text().strip()
        if manual:
            paths += [p.strip() for p in manual.split(",") if p.strip()]
        if not paths:
            QMessageBox.information(self, "没有选择文件",
                                    "请在列表里选择，或在输入框里填写路径。")
            return
        checkpoint = self.checkpoint.currentData()
        if not checkpoint:
            QMessageBox.information(self, "没有可用的检查点", "请先训练一个模型。")
            return
        self.ctx.run("predict", {
            "paths": paths, "checkpoint": checkpoint,
            "max_models": self.max_models.value(),
            "threads": self.threads.value(),
            "verify": self.verify.isChecked(),
        })

    def load_results(self, silent: bool = False) -> None:
        import json

        path = os.path.join(self.ctx.dirs["outputs"], "predictions.json")
        if not os.path.exists(path):
            if silent:
                self.notify("没有可读的 predictions.json。")
            else:
                QMessageBox.information(self, "还没有结果", "请先跑一次预测。")
            return
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        rows.sort(key=lambda r: r.get("predicted_relative_energy_kcal_per_mol") or 1e9)
        fill_table(self.results, [[
            r.get("name"), r.get("model_index"),
            num(r.get("predicted_relative_energy_kcal_per_mol"), 3),
            num(r.get("true_relative_energy_kcal_per_mol"), 3),
            num(r.get("error_kcal_per_mol"), 3),
            r.get("n_residues"),
        ] for r in rows])
        self.results.resizeColumnsToContents()

    def update_state(self, state: dict[str, Any]) -> None:
        current = self.checkpoint.currentData()
        self.checkpoint.clear()
        for item in predict_checkpoints(self.ctx.dirs["outputs"]):
            self.checkpoint.addItem(item["name"], item["path"])
        if not self.checkpoint.count():
            self.checkpoint.addItem("（无可用检查点，请先训练）", None)
        elif current:
            index = self.checkpoint.findData(current)
            if index >= 0:
                self.checkpoint.setCurrentIndex(index)

        if not self.files.count():
            prefix = self.ctx.dirs["raw"].replace("\\", "/")
            for name in list_pdb_files(self.ctx.dirs["raw"]):
                self.files.addItem(f"{prefix}/{name}")

    def on_job_finished(self, status: str) -> None:
        if status == "done":
            self.load_results(silent=True)


# --------------------------------------------------------------------------- #
# 迭代训练
# --------------------------------------------------------------------------- #


class IteratePage(Page):
    title = "迭代训练"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.heading(
            "迭代训练",
            "每轮：标注新条目 → 重建数据集 → 从上一轮热启动 → 评估 → 记录。"
            "切分在第 1 轮冻结，之后新增的蛋白只进 train，所以跨轮指标可比。"
            "build-limit 用来分批，避免一轮就把几百条全标了。",
        )

        self.rounds = spin(1, 1, 1000, tip="跑几轮「加数据→重训→评估→记录」")
        self.limit = spin(20, 0, 100000, step=10,
                          tip="本轮最多标注多少条新条目。填 0 表示全部"
                              "（几百条会跑几十小时）")
        self.max_res = spin(200, 5, 10000, step=10, tip="本轮允许标注的条目残基上限")
        self.epochs = spin(20, 1, 10000, tip="每轮训练多少轮")
        self.model = combo([("SchNet", "schnet"), ("MLP", "mlp")],
                           tip="每轮用哪种模型训练。换成更大的模型要与 --out-dir 一起用，"
                               "因为结构变了就不能热启动")
        self.threads = spin(8, 1, 64,
                            tip="OpenMM 内部线程数。实测 8 线程比 1 线程快约 3.5–4 倍")
        self.workers = spin(1, 1, 32,
                            tip="同时处理几个蛋白。workers × threads 不要超过物理核数，"
                                "否则互相抢核反而更慢")
        self.build_new = check("标注新条目", True,
                               tip="自动发现 data/raw 里还没标注的条目并打标签")
        self.warm = check("热启动", True,
                          tip="每轮从上一轮检查点热启动。关掉就是每轮从零训，可作为对照")

        box = QGroupBox("迭代设置")
        lay = QVBoxLayout(box)
        lay.addWidget(grid([
            [("轮数 rounds", self.rounds, None),
             ("本轮最多标注 N 条", self.limit, None),
             ("最大残基数 max_residues", self.max_res, None),
             ("每轮训练轮数", self.epochs, None)],
            [("模型", self.model, None), ("OpenMM 线程数", self.threads, None),
             ("并行进程数", self.workers, None), ("", QWidget(), None)],
        ]))
        row = QHBoxLayout()
        row.addWidget(self.build_new)
        row.addWidget(self.warm)
        row.addWidget(button("运行迭代", self.do_iterate, primary=True,
                             tip="启动多轮迭代训练"))
        row.addWidget(button("读取轮次表", self.load_rounds))
        row.addStretch(1)
        lay.addLayout(row)
        self.root.addWidget(box)

        self.rounds_table = table(["轮次", "蛋白数", "训练/验证/测试", "验证 MAE",
                                   "测试 MAE", "测试 ρ", "组内 ρ", "常数基线", "跨度"])
        self.root.addWidget(self.rounds_table, 1)

    def do_iterate(self) -> None:
        self.ctx.run("iterate", {
            "rounds": self.rounds.value(),
            "build_limit": self.limit.value(),
            "max_residues": self.max_res.value(),
            "epochs": self.epochs.value(),
            "model": self.model.currentData(),
            "threads": self.threads.value(),
            "workers": self.workers.value(),
            "build_new": self.build_new.isChecked(),
            "warm_start": self.warm.isChecked(),
        })

    def load_rounds(self, silent: bool = False) -> None:
        import json

        path = os.path.join(self.ctx.dirs["outputs"], "iterative", "rounds.jsonl")
        if not os.path.exists(path):
            if silent:
                self.notify("还没有轮次记录。")
            else:
                QMessageBox.information(self, "还没有轮次记录", "请先跑一次迭代训练。")
            return
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        span = lambda r: ("—" if r.get("prediction_span_ratio") is None
                          else f'{100 * r["prediction_span_ratio"]:.0f}%')
        fill_table(self.rounds_table, [[
            r.get("round"), r.get("n_proteins"),
            f'{r.get("n_train")}/{r.get("n_val")}/{r.get("n_test")}',
            num(r.get("val_mae")), num(r.get("test_mae")),
            num(r.get("test_spearman"), 3), num(r.get("within_protein_rho"), 3),
            num(r.get("constant_baseline_mae")), span(r),
        ] for r in rows])

    def on_job_finished(self, status: str) -> None:
        if status == "done":
            self.load_rounds(silent=True)


# --------------------------------------------------------------------------- #
# 帮助
# --------------------------------------------------------------------------- #

HELP_HTML = """
<h2>帮助 / 常见问题</h2>
<p style="color:#666">这里回答使用中最常遇到的疑问，以及这个项目<b>真实的能力边界</b>——
有些问题的答案是「现在还做不到」，那也如实写在这里。</p>

<h3>一、这个项目在做什么</h3>
<p><b>这个模型预测什么？</b><br>
输入一个蛋白质构象（PDB 坐标 + 元素类型），输出它的<b>相对构象能</b>
ΔE = E − min(E)，单位 kcal/mol。绝对能量主要由原子组成和长程溶剂化决定，
换个蛋白就完全不可比；真正有意义的是「这个构象比该蛋白自己最舒服的构象差多少」。</p>
<p>标签（真值）不是实验数据，而是 <b>AMBER14 + GBn2 隐式溶剂</b>力场算出来的能量。
所以模型的上限就是力场的上限——它是一台力场的快速代理，不是新的物理。</p>

<p><b>现在能当打分函数用吗？</b><br><b>还不能。</b>在 14 个蛋白、730 个构象上实测：
测试 MAE 121 kcal/mol，而「永远输出平均值」的常数基线是 107；预测跨度只有真实范围的
<b>2%</b>（说明基本在输出常数）；组内排序相关性只有 0.17。它确实带一点信号
（全局 Spearman 0.43），但不足以用来挑构象。请把它当作<b>一条跑通的流水线</b>。</p>

<h3>二、指标怎么读（最容易骗自己的部分）</h3>
<p><b>为什么说 MAE 单独看会骗人？</b><br>
一个几乎只输出常数的模型，只要那个常数稍微好一点，MAE 就能「赢过常数基线」。
本项目真实发生过：把径向基加密后 val MAE 从 129.5 降到 114.3（首次超过基线 118.3），
但 <b>Spearman 从 0.430 掉到 −0.036</b>，预测跨度仍只有 2%。
它是靠「把预测压得更接近均值」赢的 MAE。<br>
<b>所以务必同时看三个量</b>：MAE、预测跨度、组内 ρ。</p>

<table border="0" cellpadding="4">
<tr><th align="left">指标</th><th align="left">含义</th><th align="left">陷阱</th></tr>
<tr><td>MAE</td><td>平均绝对误差 (kcal/mol)</td><td>直观，但对大误差不敏感</td></tr>
<tr><td>RMSE</td><td>均方根误差</td><td>离群点就能拉高；与 MAE 差距大说明有离群点</td></tr>
<tr><td>R²</td><td>解释了多少方差</td><td>会被「认出这是哪个蛋白」撑高</td></tr>
<tr><td>Spearman ρ</td><td>排序一致性</td><td>全局值会被「区分不同蛋白」撑高</td></tr>
<tr><td>组内 ρ</td><td>同一蛋白内部的排序相关性</td><td><b>这才是打分函数真正需要的</b></td></tr>
<tr><td>预测跨度</td><td>(预测max−min)/(真实max−min)</td><td>只有百分之几 = 在输出常数</td></tr>
<tr><td>常数基线</td><td>永远输出训练集平均值</td><td>任何模型都必须显著优于它</td></tr>
</table>

<p><b>为什么必须按蛋白质切分？</b><br>
同一蛋白的相邻 MD 快照能量几乎一样。按帧随机切分会让测试集里出现训练集的近似副本，
指标虚高。本项目直接量过：按帧切分时 <b>97.3% 的测试帧最近邻来自同一蛋白，
52.7% 是近似重复</b>；按蛋白质切分时都是 0%。</p>

<h3>三、数据与时间</h3>
<p><b>加更多数据能提升精度吗？</b>在冻结切分下按 25% / 50% / 100% 取子集：
81 个构象 → 验证 MAE 128.56；160 个 → <b>143.02</b>；323 个 → 127.16；
常数基线 118.28。<b>数据翻倍反而变差，三点全在基线之上</b>，而且训练损失在任何数据量下
都是 0.30–0.31。这是「学到的东西不迁移」，不是「样本不够」。加轮数也一样：
4 轮和 20 轮结果相同。<br>
所以现阶段<b>不要</b>花几十小时去标几百个蛋白——先把目标设计修好。</p>

<p><b>打标签 / 训练要多久？</b>打标签每蛋白 2–4 分钟（8 线程），100 个蛋白几小时；
扫描 2741 个文件约 6 分钟；训练 323 个构象、SchNet、batch 16 约 100 秒/轮。
打标签和扫描都可中断后重跑，已完成的会跳过。</p>

<p><b>磁盘和内存？</b>已标注数据约 5 KB/构象（415 个蛋白 ≈ 122 MB）；
图缓存约 <b>0.5 MB/构象且全进内存</b>，2 万构象 ≈ 10 GB，超过就勾「禁用图缓存」；
原始 PDB 2741 个约 1.66 GB。</p>

<h3>四、操作与排错</h3>
<p><b>threads 和 workers 怎么配？</b>threads 是 OpenMM 内部线程数，workers 是同时处理几个
蛋白。<code>workers × threads</code> 不要超过物理核数，否则互相抢核更慢。
4 核 8 线程建议 <code>--workers 4 --threads 2</code>。</p>

<p><b>为什么改了模型结构就不能热启动？</b>hidden_dim / n_interactions / cutoff / n_rbf
任一变化都会让权重形状对不上。最坏结果不是报错，而是<b>只加载了一部分张量、
其余保持随机</b>——得到一个看着能跑、实际半随机的模型。所以程序在加载前逐字段比对配置，
不一致就直接报错并指出是哪个字段变了。</p>

<p><b>热启动和续训的区别？</b>热启动：用旧权重，但优化器、轮次计数、历史全部重来
（数据变多时用）。续训：连 AdamW 的两个动量和轮次计数一起恢复（长训练被中断时用）。
热启动默认<b>沿用检查点里的归一化</b>，否则输出层会按错误尺度标定。</p>

<p><b>排序不对正常吗？</b>正常。组内 ρ 只有 0.17，排错顺序并不奇怪。
用「对照真值」可以直接看到误差。</p>

<p><b>怎么复现某一次运行？</b>界面只负责拼命令行，逻辑全在 CLI 里。日志里显示的那行命令
可以直接粘到终端重跑。每个运行目录里还有 config.json、history.json、
data_summary.json，检查点里带着归一化统计量和切分清单。</p>

<p><b>报错 No module named 'pdbenergy'？</b>包不在 Python 搜索路径里。
在项目根目录下运行，或重装：<code>pip install -e .</code>。
目录被改名或移动过时必须重装（editable 安装会静默失效）。</p>

<p><b>提示缺少 OpenMM / PDBFixer？</b>只有打标签、预测（要加氢）、verify 需要它们。
安装：<code>pip install -e ".[physics]"</code>。</p>

<h3>五、术语表</h3>
<table border="0" cellpadding="3">
<tr><td><b>构象</b></td><td>不改变化学键、只绕单键旋转得到的分子形状。</td></tr>
<tr><td><b>残基</b></td><td>蛋白质链上的一个氨基酸单元。46 残基的蛋白约 640 个原子（含氢）。</td></tr>
<tr><td><b>打标签</b></td><td>用物理力场算出每个构象的能量，作为神经网络的学习目标。</td></tr>
<tr><td><b>力场</b></td><td>用经验函数近似能量随坐标变化的模型。本项目用 AMBER14 + GBn2。</td></tr>
<tr><td><b>隐式溶剂</b></td><td>把水当连续介质，而不是显式加几千个水分子。</td></tr>
<tr><td><b>二面角旋转</b></td><td>绕可旋转单键旋转一侧子树，精确保持所有键长键角。</td></tr>
<tr><td><b>能量最小化</b></td><td>求局部极小值（L-BFGS）。用来定义相对能量的零点。</td></tr>
<tr><td><b>能量均分</b></td><td>温度 T 下势能平均比极小值高约 ½N·kT。640 原子在 300 K 约高 570 kcal/mol。</td></tr>
<tr><td><b>图神经网络</b></td><td>用「原子=节点、邻近关系=边」表示分子，通过消息传递更新原子特征。</td></tr>
<tr><td><b>SchNet</b></td><td>一种消息传递网络，边权由原子间距离经 RBF 展开后决定。</td></tr>
<tr><td><b>径向基 RBF</b></td><td>把标量距离展开成一组高斯函数值。本项目间距 0.161 Å。</td></tr>
<tr><td><b>cutoff</b></td><td>只让距离小于它的原子对交换消息。越小越快，但截断长程物理。</td></tr>
<tr><td><b>求和池化</b></td><td>把每个原子的能量贡献相加得到总能量，保证能量是广延量。</td></tr>
<tr><td><b>归一化</b></td><td>目标减均值除标准差。必须用训练集统计量，且存进检查点。</td></tr>
<tr><td><b>Huber 损失</b></td><td>误差小时二次、大时线性，防止高能离群点主导梯度。</td></tr>
<tr><td><b>数据泄漏</b></td><td>测试集含训练集的近似副本，导致指标虚高。本项目实测 53%。</td></tr>
<tr><td><b>检查点</b></td><td>权重 + 配置 + 归一化统计量 + 切分清单。自解释，可单独推理。</td></tr>
<tr><td><b>图缓存</b></td><td>把「结构→图」的结果存内存复用。约 0.5 MB/构象，是内存主要消耗。</td></tr>
</table>

<h3>六、下一步该改什么</h3>
<p>按实测证据排序（完整论述见 TeachFlow.md §12.3）：</p>
<ol>
<li><b>目标设计（最该动的地方）。</b>现在的 ΔE 把热激发幅度（MD 帧比极小值高
300–1000 kcal/mol，随蛋白大小线性增长）和构象形变（二面角帧通常 0–100 kcal/mol）
混在一起。可试：按原子数归一的目标 ΔE/N、砍掉 450 K 高温尾巴、或分开建模。</li>
<li><b>特征分辨率。</b>基函数间距 0.161 Å 比键长热涨落（0.05–0.1 Å）还粗，
加密到 64–96 后 MAE 降了 12–16%，但代价是排序能力塌掉。有用，但不是全部答案。</li>
<li><b>排序损失。</b>用 pairwise ranking loss 直接优化组内 ρ。</li>
<li><b>力监督。</b>把能量梯度也加进损失，这是机器学习势函数的标准做法。</li>
<li><b>最后才是加数据。</b>等目标修好、指标越过常数基线之后，数据量大概率会重新变成瓶颈。</li>
</ol>
"""


class HelpPage(Page):
    title = "帮助"

    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(HELP_HTML)
        self.root.addWidget(self.browser, 1)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    PAGES = (OverviewPage, DataPage, TrainPage, EvalPage, PredictPage,
             IteratePage, HelpPage)

    def __init__(self, ctx: AppContext):
        super().__init__()
        ctx.window = self
        self.ctx = ctx
        self.active_job_id: str | None = None
        self.cursor = 0

        self.setWindowTitle("PDBEnergy — 训练 / 预测 / 测试")
        self.resize(1280, 860)

        # ---- 左侧导航 + 右侧页面 ----------------------------------------- #
        self.nav = QListWidget()
        self.nav.setMaximumWidth(200)
        self.nav.setMinimumWidth(160)
        self.stack = QStackedWidget()
        self.pages: list[Page] = []
        for cls in self.PAGES:
            page = cls(ctx)
            self.pages.append(page)
            self.nav.addItem(cls.title)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.nav)
        top.addWidget(self.stack)
        top.setStretchFactor(1, 1)

        # ---- 底部：命令行预览 + 实时日志（都用等宽字体）------------------ #
        self.command = QPlainTextEdit()
        self.command.setReadOnly(True)
        self.command.setMaximumHeight(52)
        self.command.setFont(monospace_font(9))
        self.command.setPlaceholderText("启动任务后这里显示它实际执行的命令行（可直接复制到终端重跑）")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(monospace_font(9))
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("实时日志：任务的标准输出会逐行出现在这里")
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.job_label = QLabel("无任务")
        self.job_label.setStyleSheet("color:#666;")

        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(8, 6, 8, 8)
        blay.addWidget(self.job_label)
        blay.addWidget(self.command)
        blay.addWidget(self.log, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        # ---- 工具栏 / 状态栏 --------------------------------------------- #
        refresh_action = QAction("刷新", self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self.refresh_state)
        self.cancel_action = QAction("取消任务", self)
        self.cancel_action.setEnabled(False)
        self.cancel_action.triggered.connect(self.cancel_job)
        open_action = QAction("打开输出目录", self)
        open_action.triggered.connect(self.open_outputs)
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.addAction(refresh_action)
        toolbar.addAction(self.cancel_action)
        toolbar.addAction(open_action)

        self.setStatusBar(QStatusBar())
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

        # ---- 轮询 --------------------------------------------------------- #
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.poll_job)
        self.timer.start()

        self.refresh_state()
        self.check_environment()

    # -- 启动自检 ----------------------------------------------------------- #
    def check_environment(self) -> None:
        """把关键路径写进日志，并就地做一次导入自检。

        就地检查（``importlib.util.find_spec``）而不是起一个子进程：启动瞬间不该
        为一次 ``--help`` 级的开销卡顿，也避免自检进程占住工作目录。真正的子进程
        级诊断留给任务失败时——``JobManager`` 会在日志里给出完整提示。
        """
        mgr = self.ctx.manager
        root_ok = os.path.isdir(mgr.project_root)
        self.log_message(
            f"[自检] 解释器     {mgr.python}\n"
            f"[自检] 项目根目录 {mgr.project_root}（存在：{root_ok}）\n"
            f"[自检] 工作目录   {mgr.cwd}\n"
            f"[自检] 数据目录   raw={self.ctx.dirs['raw']}"
        )
        problem = None
        if not root_ok:
            problem = f"项目根目录不存在：{mgr.project_root}"
        else:
            try:
                if importlib.util.find_spec("pdbenergy.cli") is None:
                    problem = "找不到模块 pdbenergy.cli"
            except (ImportError, ValueError) as exc:
                problem = f"{type(exc).__name__}: {exc}"
        if problem is None:
            self.log_message("[自检] ✓ 能找到 pdbenergy.cli，环境正常。")
            return
        self.log_message(f"[自检] ✗ {problem} —— 现在启动任务很可能直接失败。")
        for line in mgr.import_hint("pdbenergy.cli", problem).splitlines():
            self.log_message(line)

    # -- 状态 --------------------------------------------------------------- #
    def refresh_state(self) -> None:
        state = self.ctx.refresh()
        for page in self.pages:
            try:
                page.update_state(state)
            except Exception as exc:                     # 一个页面出错不该拖垮界面
                self.log_message(f"[界面] {page.title} 刷新失败：{exc}")
        counts = state.get("counts", {})
        invent = state.get("inventory")
        self.status_label.setText(
            f"raw {counts.get('raw_pdb', 0)} · 已标注 {counts.get('labelled', 0)} · "
            f"待标注 {counts.get('pending', 0)} · 体检 "
            + (f"{invent['usable']}/{invent['scanned']} 可用" if invent else "未扫描")
        )

    def open_outputs(self) -> None:
        path = os.path.abspath(self.ctx.dirs["outputs"])
        os.makedirs(path, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)                            # noqa: S606
        else:
            QMessageBox.information(self, "输出目录", path)

    # -- 任务 --------------------------------------------------------------- #
    def start_job(self, label_text: str, module: str, args: list[str]) -> None:
        if self.active_job_id:
            QMessageBox.information(self, "已有任务在运行",
                                    "请等它结束，或先点「取消任务」。")
            return
        job = self.ctx.manager.start(label_text, args, module=module)
        self.active_job_id = job.id
        self.cursor = 0
        self.log.clear()
        self.job_label.setText(f"运行中：{label_text}")
        self.command.setPlainText(job.public(0)["command"])
        self.cancel_action.setEnabled(True)
        self.log_message(f"$ {job.public(0)['command']}")

    def poll_job(self) -> None:
        if not self.active_job_id:
            return
        job = self.ctx.manager.get(self.active_job_id)
        if job is None:
            self.active_job_id = None
            return
        data = job.public(self.cursor)
        if data["dropped"]:
            self.log_message(f"[界面] {data['dropped']} 行较早的输出已被丢弃")
        for line in data["lines"]:
            self.log.appendPlainText(line)
        self.cursor = data["cursor"]
        self.job_label.setText(
            f"{job.label} — {data['status']} {data['elapsed']:.0f} 秒"
        )
        if data["status"] != "running":
            status = data["status"]
            self.active_job_id = None
            self.cancel_action.setEnabled(False)
            self.log_message(
                {"done": "✓ 任务完成", "failed": "✗ 任务失败（见上方输出）",
                 "cancelled": "已取消"}.get(status, status)
            )
            self.refresh_state()
            for page in self.pages:
                try:
                    page.on_job_finished(status)
                except Exception as exc:
                    self.log_message(f"[界面] {page.title} 收尾失败：{exc}")

    def cancel_job(self) -> None:
        if self.active_job_id:
            self.ctx.manager.cancel(self.active_job_id)

    def log_message(self, text: str) -> None:
        self.log.appendPlainText(text)

    # -- 关闭：先杀子任务 --------------------------------------------------- #
    def closeEvent(self, event) -> None:                  # noqa: N802
        """子进程不会随父进程一起死，必须显式终止。

        不这么做，关掉窗口后 OpenMM 打标签可能继续占满所有核心跑几小时。
        """
        stopped = self.ctx.manager.shutdown()
        if stopped:
            print(f"terminated {stopped} running job(s)")
        event.accept()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="pdbenergy gui",
        description="PDBEnergy 桌面界面（PySide6）：数据 / 训练 / 预测 / 测试。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="JSON 配置，用作默认值")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("PDBEnergy")
    # 正文 Times New Roman（中文自动回退到宋体）；代码区由 monospace_font() 单独设置。
    body = QFont("Times New Roman", 10)
    body.setStyleHint(QFont.Serif)
    app.setFont(body)

    cfg = Config.load(args.config) if args.config else Config()
    ctx = AppContext(
        cfg=cfg,
        dirs={"raw": resolve_dir(args.raw_dir), "interim": resolve_dir(args.interim_dir),
              "processed": resolve_dir(args.processed_dir),
              "outputs": resolve_dir(args.outputs_dir)},
        # No argument: JobManager falls back to the project root, so a shortcut
        # started from anywhere still finds data/ and outputs/.
        manager=JobManager(),
    )
    window = MainWindow(ctx)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
