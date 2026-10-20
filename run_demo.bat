@echo off
REM Double-click to reproduce every result and map, then open demo_report.html.
REM Options are passed to run_demo.py, e.g.  run_demo.bat --quick   (fast demo, about 3 minutes)
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
if exist "%USERPROFILE%\anaconda3\Library\bin" set "PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Library\bin;%PATH%"
set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if not defined PY set PY=python
echo Using %PY%
"%PY%" run_demo.py %*
set RESULT=%ERRORLEVEL%
if not defined NOPAUSE (
  if exist demo_report.html start "" demo_report.html
  pause
)
exit /b %RESULT%
