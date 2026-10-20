@echo off
REM One-time setup: creates a private Python environment (.venv) in this folder and installs the
REM packages from requirements.txt. Nothing outside this folder is changed.
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set BASE=python
if exist "%USERPROFILE%\anaconda3\python.exe" (
  set "BASE=%USERPROFILE%\anaconda3\python.exe"
  REM Anaconda's Python needs its DLL folders on PATH (SSL for pip) when conda is not activated
  set "PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Library\bin;%USERPROFILE%\anaconda3\Scripts;%PATH%"
)
if exist ".venv\Scripts\python.exe" if not exist ".venv\Scripts\pip.exe" (
  echo Removing an incomplete .venv from an earlier failed setup
  rmdir /s /q .venv
)
if exist ".venv\Scripts\python.exe" (
  echo .venv already exists; updating packages
) else (
  echo Creating .venv with %BASE%
  "%BASE%" -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
echo.
echo Setup complete. Next: double-click run_demo.bat
if not defined NOPAUSE pause
exit /b 0
:fail
echo Setup failed - see the messages above.
if not defined NOPAUSE pause
exit /b 1
