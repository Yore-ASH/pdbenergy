@echo off
REM Launch the PySide6 desktop GUI from the project root with a clean Python
REM environment.
REM
REM Why this exists: a job launched by the GUI once died with
REM     No module named pdbenergy.cli
REM while the interpreter path, the project root and the working directory the
REM GUI printed were all correct.  The one thing left that can do that is an
REM inherited PYTHON* variable (PYTHONHOME above all): it makes the venv
REM interpreter look for its standard library in another installation, and a
REM desktop shortcut or an IDE can carry such a variable in silently.
REM
REM Clearing them here costs nothing and removes that whole class of failure.
REM Double-click this file, or point a shortcut at it.  The working directory no
REM longer matters: the GUI anchors its data paths to the project root.
REM
REM This file is deliberately ASCII-only.  cmd.exe reads .cmd files in the OEM
REM code page, so UTF-8 comments get mangled into stray commands.

setlocal
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSAFEPATH="
set "PYTHONSTARTUP="
set "PYTHONEXECUTABLE="
set "PYTHONNOUSERSITE="
set "PYTHONUSERBASE="
set "__PYVENV_LAUNCHER__="

cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo [start_gui] .venv\Scripts\python.exe not found.
    echo [start_gui] Create the environment first:
    echo [start_gui]     python -m venv .venv
    echo [start_gui]     .venv\Scripts\python.exe -m pip install -e ".[gui,physics]"
    exit /b 1
)

".venv\Scripts\python.exe" -m pdbenergy.gui_qt %*
exit /b %ERRORLEVEL%
