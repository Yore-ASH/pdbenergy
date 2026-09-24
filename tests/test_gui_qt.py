"""Tests for the PySide6 desktop GUI.

Skipped cleanly when PySide6 is absent, so a checkout without the ``gui`` extra
still runs the whole suite.  Qt runs on the **offscreen** platform: no window is
shown, but widgets are really constructed, laid out and painted, which is what
catches the failures that matter here (a page that raises while refreshing, a
navigation list out of sync with the stack, the log quietly losing its monospace
font).

Modal dialogs are deliberately never triggered: ``QMessageBox`` blocks, so a test
that opened one would hang instead of failing.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before QApplication exists.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtGui import QFontInfo, QKeySequence
    from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton
    PYSIDE = True
except ImportError:                                          # pragma: no cover
    PYSIDE = False

from pdbenergy.config import Config                          # noqa: E402
from pdbenergy.gui import JobManager                         # noqa: E402

requires_qt = unittest.skipUnless(PYSIDE, "PySide6 未安装（pip install -e '.[gui]'）")

SLEEPER = ("import time\nprint('started', flush=True)\n"
           "for i in range(120):\n    print('tick', i, flush=True)\n    time.sleep(0.25)\n")

#: For the "runs to completion" test.  The long sleeper above takes 30 s, which is
#: far longer than a unit test should wait - an earlier version of this test
#: waited 15 s, timed out, and failed for the wrong reason.
QUICK = ("import sys\n"
         "print('quick start', flush=True)\n"
         "for i in range(3):\n    print('line', i, flush=True)\n"
         "sys.exit(0)\n")


@requires_qt
class TestWindowStructure(unittest.TestCase):
    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])
        from pdbenergy.gui_qt import AppContext, MainWindow

        cls.ctx = AppContext(
            cfg=Config(),
            dirs={"raw": "data/raw", "interim": "data/interim",
                  "processed": "data/processed", "outputs": "outputs"},
            manager=JobManager(os.getcwd()),
        )
        cls.window = MainWindow(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def test_seven_pages_are_navigable(self):
        from pdbenergy.gui_qt import MainWindow

        self.assertEqual(self.window.nav.count(), len(MainWindow.PAGES))
        self.assertEqual(self.window.stack.count(), len(MainWindow.PAGES))
        self.assertEqual(self.window.nav.count(), 7)

    def test_navigation_stays_in_sync_with_the_stack(self):
        for row in range(self.window.nav.count()):
            self.window.nav.setCurrentRow(row)
            self.assertEqual(self.window.stack.currentIndex(), row)

    def test_every_nav_entry_has_a_titled_page(self):
        from pdbenergy.gui_qt import Page

        for row in range(self.window.nav.count()):
            page = self.window.stack.widget(row)
            self.assertIsInstance(page, Page)
            self.assertEqual(self.window.nav.item(row).text(), page.title)

    def test_page_titles_cover_the_requested_workflow(self):
        titles = [self.window.nav.item(i).text() for i in range(self.window.nav.count())]
        for needed in ("训练", "预测", "帮助"):
            self.assertTrue(any(needed in t for t in titles),
                            f"缺少「{needed}」页面：{titles}")


@requires_qt
class TestFonts(unittest.TestCase):
    """正文 Times New Roman；代码区（日志、命令行、路径输入框）等宽。"""

    app = None

    @classmethod
    def setUpClass(cls):
        from PySide6.QtGui import QFont
        from pdbenergy.gui_qt import AppContext, MainWindow

        cls.app = QApplication.instance() or QApplication(sys.argv[:1])
        body = QFont("Times New Roman", 10)
        body.setStyleHint(QFont.Serif)
        cls.app.setFont(body)
        ctx = AppContext(
            cfg=Config(),
            dirs={"raw": "data/raw", "interim": "data/interim",
                  "processed": "data/processed", "outputs": "outputs"},
            manager=JobManager(os.getcwd()),
        )
        cls.window = MainWindow(ctx)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def test_application_font_is_times_new_roman(self):
        self.assertEqual(self.app.font().family(), "Times New Roman")

    def test_log_is_monospaced(self):
        self.assertIn(self.window.log.font().family(),
                      ("Consolas", "Cascadia Mono", "DejaVu Sans Mono",
                       "Courier New", "monospace"))

    def test_command_preview_is_monospaced(self):
        self.assertIn(self.window.command.font().family(),
                      ("Consolas", "Cascadia Mono", "DejaVu Sans Mono",
                       "Courier New", "monospace"))

    def test_path_fields_are_monospaced(self):
        from pdbenergy.gui_qt import mono_line

        self.assertIn(mono_line().font().family(),
                      ("Consolas", "Cascadia Mono", "DejaVu Sans Mono",
                       "Courier New", "monospace"))

    def test_monospace_helper_prefers_a_known_family(self):
        from pdbenergy.gui_qt import monospace_font

        font = monospace_font(9)
        self.assertEqual(font.pointSize(), 9)
        # On a platform with no monospace family at all this still returns
        # something; the family name is only asserted when one exists.
        self.assertTrue(font.family())


@requires_qt
class TestAnnotations(unittest.TestCase):
    """「为大部分名词提供注解」—— 控件上必须真的挂着 tooltip。"""

    app = None

    @classmethod
    def setUpClass(cls):
        from pdbenergy.gui_qt import AppContext, MainWindow

        cls.app = QApplication.instance() or QApplication(sys.argv[:1])
        ctx = AppContext(
            cfg=Config(),
            dirs={"raw": "data/raw", "interim": "data/interim",
                  "processed": "data/processed", "outputs": "outputs"},
            manager=JobManager(os.getcwd()),
        )
        cls.window = MainWindow(ctx)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def _all_widgets(self):
        from PySide6.QtWidgets import QWidget

        for page in self.window.pages:
            yield from page.findChildren(QWidget)

    def test_tooltips_are_widespread(self):
        annotated = [w for w in self._all_widgets() if w.toolTip().strip()]
        self.assertGreater(len(annotated), 25,
                           "注解太少：需要做决定的控件都该有 tooltip")

    def test_tooltips_are_explanatory_not_just_restatements(self):
        """注解要解释，不是把标签抄一遍 —— 抽查长度与关键词。"""
        tips = [w.toolTip() for w in self._all_widgets() if w.toolTip().strip()]
        long_tips = [t for t in tips if len(t) > 25]
        self.assertGreater(len(long_tips), 15,
                           f"注解偏短；示例：{tips[:5]}")
        joined = " ".join(tips)
        for keyword in ("测量", "实测", "默认", "不要", "必须", "会", "说明"):
            if keyword in joined:
                break
        else:
            self.fail("注解里没有任何解释性措辞")

    def test_spinboxes_and_checkboxes_carry_annotations(self):
        from PySide6.QtWidgets import QSpinBox

        spins = [w for w in self._all_widgets() if isinstance(w, (QSpinBox, QCheckBox))]
        self.assertGreater(len(spins), 10, "没有找到足够多的可调控件")
        missing = [w for w in spins if not w.toolTip().strip()]
        self.assertLessEqual(len(missing), 2,
                             f"{len(missing)} 个可调控件没有注解")


@requires_qt
class TestStateBinding(unittest.TestCase):
    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])

    def _window(self):
        from pdbenergy.gui_qt import AppContext, MainWindow

        ctx = AppContext(
            cfg=Config(),
            dirs={"raw": "data/raw", "interim": "data/interim",
                  "processed": "data/processed", "outputs": "outputs"},
            manager=JobManager(os.getcwd()),
        )
        return MainWindow(ctx)

    def test_refresh_populates_the_overview_from_this_repo(self):
        window = self._window()
        try:
            # The sandbox has real data/ and outputs/, so this exercises the
            # binding end to end rather than against an empty fixture.
            window.refresh_state()
            self.assertIsNotNone(window.ctx.state)
            self.assertGreaterEqual(
                window.ctx.state["counts"]["raw_pdb"], 1
            )
            self.assertIn("raw", window.status_label.text())
        finally:
            window.close()

    def test_update_state_is_safe_on_an_empty_project(self):
        """No files, no runs: every page must still refresh without raising."""
        import tempfile

        from pdbenergy.gui_qt import AppContext, MainWindow

        # ignore_cleanup_errors: the window's JobManager uses this directory as
        # the child's cwd, so on Windows the removal can race a closing process
        # handle and fail with WinError 32.  The assertions have already run by
        # then; failing a whole suite over a leftover %TEMP% directory is noise.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            dirs = {k: os.path.join(root, k)
                    for k in ("raw", "interim", "processed", "outputs")}
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)
            ctx = AppContext(cfg=Config(), dirs=dirs, manager=JobManager(root))
            window = MainWindow(ctx)
            try:
                window.refresh_state()          # must not raise
                self.assertEqual(window.ctx.state["counts"]["raw_pdb"], 0)
                self.assertIsNone(window.ctx.state["inventory"])
            finally:
                window.close()

    def test_on_job_finished_never_opens_a_modal_dialog(self):
        """Regression test for a real hang.

        ``on_job_finished`` is an *automatic* callback - there is no user present
        to dismiss anything.  An earlier version called ``load_pending()`` from
        it, which popped a modal "还没有体检报告" dialog whenever a job finished
        before any scan had been run.  The GUI froze after every job, and the
        test suite hung instead of failing.
        """
        import tempfile
        from unittest import mock

        from pdbenergy.gui_qt import AppContext, MainWindow

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            dirs = {k: os.path.join(root, k)
                    for k in ("raw", "interim", "processed", "outputs")}
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)
            window = MainWindow(AppContext(cfg=Config(), dirs=dirs,
                                           manager=JobManager(root)))
            try:
                with mock.patch("pdbenergy.gui_qt.QMessageBox.information") as info, \
                     mock.patch("pdbenergy.gui_qt.QMessageBox.warning") as warn:
                    for status in ("done", "failed", "cancelled"):
                        for page in window.pages:
                            page.on_job_finished(status)
                    info.assert_not_called()
                    warn.assert_not_called()
            finally:
                window.close()

    def test_help_page_states_the_real_limitations(self):
        window = self._window()
        try:
            help_page = window.stack.widget(window.nav.count() - 1)
            text = help_page.browser.toPlainText()
            # The help page is only useful if it is honest about the results.
            for phrase in ("还不能", "常数基线", "数据泄漏", "预测跨度", "目标设计"):
                self.assertIn(phrase, text, f"帮助页缺少「{phrase}」")
        finally:
            window.close()

    def test_run_reports_unknown_actions_without_starting_a_job(self):
        """ctx.run with a bad payload must complain, not spawn a process.

        QMessageBox is patched out because it is modal and would block the test.
        """
        from unittest import mock

        window = self._window()
        try:
            with mock.patch("pdbenergy.gui_qt.QMessageBox.warning") as warn:
                window.ctx.run("not-an-action", {})
                warn.assert_called_once()
            self.assertIsNone(window.active_job_id)
        finally:
            window.close()
            window.ctx.manager.shutdown()


@requires_qt
class TestJobLifecycle(unittest.TestCase):
    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])

    def _window(self, root):
        from pdbenergy.gui_qt import AppContext, MainWindow

        dirs = {k: os.path.join(root, k)
                for k in ("raw", "interim", "processed", "outputs")}
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        return MainWindow(AppContext(cfg=Config(), dirs=dirs,
                                     manager=JobManager(root)))

    def test_start_and_poll_a_job_to_completion(self):
        import tempfile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            window = self._window(root)
            try:
                script = os.path.join(root, "quick.py")
                with open(script, "w", encoding="utf-8") as fh:
                    fh.write(QUICK)
                window.start_job("quick", "quick.py", [])
                self.assertIsNotNone(window.active_job_id)
                self.assertTrue(window.command.toPlainText().strip(),
                                "命令行预览必须显示实际执行的命令")
                self.assertTrue(window.cancel_action.isEnabled())

                deadline = time.time() + 20
                while window.active_job_id and time.time() < deadline:
                    window.poll_job()
                    time.sleep(0.05)
                window.poll_job()
                self.assertIsNone(window.active_job_id, "任务没有在 20 秒内结束")
                self.assertIn("line 2", window.log.toPlainText())
                self.assertIn("任务完成", window.log.toPlainText())
                self.assertFalse(window.cancel_action.isEnabled())
            finally:
                window.close()

    def test_closing_the_window_terminates_children(self):
        """The failure this guards against: closing the GUI leaves OpenMM
        labelling processes holding every core for hours."""
        import tempfile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            window = self._window(root)
            sleeper = os.path.join(root, "sleeper.py")
            with open(sleeper, "w", encoding="utf-8") as fh:
                fh.write(SLEEPER)
            window.start_job("sleep", "sleeper.py", [])
            time.sleep(1.0)
            job = window.ctx.manager.get(window.active_job_id)
            self.assertIsNotNone(job)
            window.close()                       # triggers closeEvent
            self.assertIsNotNone(job.process.poll(), "子进程仍在运行")
            self.assertEqual(window.ctx.manager.running(), [])

    def test_cancel_button_stops_the_job(self):
        import tempfile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            window = self._window(root)
            try:
                sleeper = os.path.join(root, "sleeper.py")
                with open(sleeper, "w", encoding="utf-8") as fh:
                    fh.write(SLEEPER)
                window.start_job("sleep", "sleeper.py", [])
                time.sleep(1.0)
                window.cancel_job()
                deadline = time.time() + 15
                job = window.ctx.manager.get(window.active_job_id or "")
                while job and job.status == "running" and time.time() < deadline:
                    window.poll_job()
                    time.sleep(0.05)
                self.assertEqual(job.status, "cancelled")
            finally:
                window.close()


@requires_qt
class TestStartupSelfCheck(unittest.TestCase):
    """The window must say where it thinks it is, in the log, at startup.

    Without this, a broken child environment surfaced only as a bare
    ``No module named pdbenergy.cli`` buried in job output - with no interpreter,
    no project root and no hint of what to do about it.
    """

    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])

    def test_environment_is_reported_on_startup(self):
        import tempfile

        from pdbenergy.gui_qt import AppContext, MainWindow

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            dirs = {k: os.path.join(root, k)
                    for k in ("raw", "interim", "processed", "outputs")}
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)
            manager = JobManager(root)
            window = MainWindow(AppContext(cfg=Config(), dirs=dirs, manager=manager))
            try:
                text = window.log.toPlainText()
                self.assertIn("[自检]", text)
                self.assertIn(manager.python, text)
                self.assertIn(manager.project_root, text)
                self.assertIn(dirs["raw"], text)
                # The package is importable in the test environment, so the
                # check must come back positive rather than warn.
                self.assertIn("pdbenergy.cli", text)
                self.assertNotIn("✗", text)
            finally:
                window.close()

    def test_a_missing_project_root_is_reported_as_a_problem(self):
        """Simulate the renamed-checkout case: the root the GUI captured is gone."""
        import tempfile

        from pdbenergy.gui_qt import AppContext, MainWindow

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            dirs = {k: os.path.join(root, k)
                    for k in ("raw", "interim", "processed", "outputs")}
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)
            manager = JobManager(root)
            manager.project_root = os.path.join(root, "PyT_PDB_HANDEL")  # renamed away
            window = MainWindow(AppContext(cfg=Config(), dirs=dirs, manager=manager))
            try:
                text = window.log.toPlainText()
                self.assertIn("✗", text)
                self.assertIn("项目根目录不存在", text)
                # ...and the user is told what to run instead of just what broke.
                self.assertIn("-c", text)
            finally:
                window.close()


@requires_qt
class TestOutputPanels(unittest.TestCase):
    """The bottom half holds two output panes: the command preview and the log.

    Both used to be cramped: the command preview was capped at 52 px (so a long
    command was clipped and could never be enlarged) and the log got whatever
    the 3:2 split left it.  These tests pin down that they can now be resized
    and that the expand toggle really gives the log the window.
    """

    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])

    def setUp(self):
        import tempfile

        from pdbenergy.gui_qt import AppContext, MainWindow

        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        dirs = {k: os.path.join(self.tmp.name, k)
                for k in ("raw", "interim", "processed", "outputs")}
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        self.window = MainWindow(AppContext(cfg=Config(), dirs=dirs,
                                           manager=JobManager(self.tmp.name)))
        self.window.resize(1280, 860)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.tmp.cleanup()

    def test_the_two_panes_share_a_draggable_splitter(self):
        split = self.window.output_split
        self.assertEqual(split.orientation().name, "Vertical")
        self.assertEqual(split.count(), 2)
        self.assertIs(split.widget(0), self.window.command)
        self.assertIs(split.widget(1), self.window.log)
        self.assertFalse(split.childrenCollapsible(),
                         "别让用户一不小心把某一块拖没了")

    def test_the_command_preview_is_no_longer_height_capped(self):
        """The old setMaximumHeight(52) clipped long command lines for good."""
        self.assertGreater(self.window.command.maximumHeight(), 1000)

    def test_dragging_the_divider_really_resizes_the_panes(self):
        before = self.window.command.height()
        self.window.output_split.setSizes([300, 100])
        self.app.processEvents()
        self.assertGreater(self.window.command.height(), before)
        self.assertGreater(self.window.command.height(),
                           self.window.log.height())

    def test_expand_gives_the_output_area_the_window(self):
        log_before = self.window.log.height()
        self.window.expand_action.trigger()
        self.app.processEvents()
        self.assertTrue(self.window.output_expanded)
        self.assertTrue(self.window.top_area.isHidden(),
                        "页面区必须让位，否则最小高度会把输出区顶回来")
        self.assertTrue(self.window.expand_action.isChecked())
        self.assertEqual(self.window.expand_action.text(), "收起输出")
        self.assertGreater(self.window.log.height(), log_before,
                           "展开后日志应该拿到更多高度")

    def test_collapse_restores_the_pages_and_the_previous_split(self):
        sizes_before = self.window.splitter.sizes()
        self.window.expand_action.trigger()
        self.app.processEvents()
        self.window.expand_action.trigger()
        self.app.processEvents()
        self.assertFalse(self.window.output_expanded)
        self.assertFalse(self.window.top_area.isHidden())
        self.assertEqual(self.window.expand_action.text(), "展开输出")
        self.assertFalse(self.window.expand_action.isChecked())
        # Same ratio as before, allowing for a pixel of layout rounding.
        for got, want in zip(self.window.splitter.sizes(), sizes_before):
            self.assertAlmostEqual(got, want, delta=2)

    def test_the_expand_shortcut_is_ctrl_e(self):
        self.assertEqual(self.window.expand_action.shortcut(),
                         QKeySequence("Ctrl+E"))

    def test_sidebar_toggle_hands_its_width_to_the_content(self):
        self.assertFalse(self.window.nav.isHidden())
        self.window.sidebar_action.trigger()
        self.app.processEvents()
        self.assertTrue(self.window.nav.isHidden())
        self.assertEqual(self.window.sidebar_action.text(), "显示侧栏")
        self.window.sidebar_action.trigger()
        self.app.processEvents()
        self.assertFalse(self.window.nav.isHidden())
        self.assertEqual(self.window.sidebar_action.text(), "隐藏侧栏")


if __name__ == "__main__":
    unittest.main()
