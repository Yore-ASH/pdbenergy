"""Reproduce, on purpose, the failure a debugged GUI produces.

Evidence so far: when the PySide6 GUI runs under VSCode's debugger, every job it
launches dies with a bare ``No module named pdbenergy.cli`` and exit code 0 -
even though the interpreter, the project root, the package directory and
``cli.py`` all exist and are readable, and the same interpreter imports the
package fine.

The suspect is debugpy: it patches ``subprocess`` so that every child of the
debuggee starts under ``pydevd``, whose bundled ``runpy`` reports a missing
module without a traceback (``pydevd_runpy.py``: ``raise error("No module named
%s")``).

This script applies *debugpy's own patch* - ``pydev_monkey``'s
``patch_new_process_functions`` - and then drives the real
:class:`pdbenergy.gui.JobManager`.  If the diagnosis is right, the same
``--help`` job that succeeds normally fails here with that bare message.

Run it, then run it with ``--unpatched`` to see the control case.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEBUGPY_PYDEVD = (
    r"c:\Users\Yore.ASH\.vscode\extensions"
    r"\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\_vendored\pydevd"
)


def patch_like_a_debugger() -> bool:
    """Apply debugpy's subprocess patch, as an attached debugger would."""
    if not os.path.isdir(DEBUGPY_PYDEVD):
        print(f"  debugpy 没找到：{DEBUGPY_PYDEVD}")
        print("  （跳过复现：本机没装 VSCode 的 Python 扩展）")
        return False
    sys.path.insert(0, DEBUGPY_PYDEVD)
    try:
        from _pydev_bundle import pydev_monkey
        pydev_monkey.patch_new_process_functions()
    except Exception as exc:
        print(f"  打补丁失败：{type(exc).__name__}: {exc}")
        return False
    print(f"  pydev_monkey.patch_new_process_functions() 已生效")
    return True


def main(argv: list[str]) -> int:
    patched = "--unpatched" not in argv
    print(f"\n{'调试器子进程补丁 ON' if patched else '调试器子进程补丁 OFF（对照组）'}")
    print("=" * 72)
    if patched and not patch_like_a_debugger():
        return 0

    from pdbenergy.gui import JobManager, running_under_debugger

    print(f"  界面看到的调试器：{running_under_debugger()}")
    manager = JobManager()
    job = manager.start("repro", ["--help"])
    # Short deadline: a hijacked child waits for a debugger that is not there, so
    # without a client it hangs rather than failing.  The takeover itself is the
    # evidence we are after.
    deadline = time.time() + 25
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    still_running = job.status == "running"
    if still_running:
        manager.cancel(job.id)
    manager.shutdown()

    text = "\n".join(job.lines)
    hijacked = "pydevd" in text
    print(f"  状态      ：{job.status}{'（卡住等调试器连接）' if still_running else ''}")
    print(f"  退出码    ：{job.returncode}")
    print("  子进程输出：")
    for line in list(job.lines)[:8]:
        print(f"      | {line}")
    print()
    if patched:
        if hijacked:
            print("  结论：✅ 复现成功 —— debugpy 把 pydevd 注入进了界面启动的子进程。")
            print("        在 VSCode 里调试器是在线的，于是被接管的子进程会用 pydevd 自带的")
            print("        runpy 解析 `python -m pdbenergy.cli`，报出一句没有 traceback 的")
            print("        'No module named pdbenergy.cli'，然后以退出码 0 结束。")
            print("        ⇒ 界面不能跑在调试器里。")
        else:
            print("  结论：补丁 ON 但子进程没被接管 —— 假设不成立，需要另找原因。")
    else:
        print("  结论：✅ 没有补丁时一切正常（对照组通过）。" if "usage" in text
              else "  结论：没有补丁也失败，问题不在调试器。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
