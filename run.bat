@echo off
REM MnemonicAi one-click launcher (Windows). Double-click, or run in a terminal.
REM   run.bat            real model (needs .\models\ornith-1.0-9b + NVIDIA)
REM   run.bat --mock     try the UI with no GPU/model
cd /d "%~dp0"

where python >nul 2>nul && (set PY=python) || (set PY=py)

if not exist mnemonicai_data\.installed (
  echo == First run: installing ==
  %PY% install.py %*
  if not exist mnemonicai_data mkdir mnemonicai_data
  echo installed> mnemonicai_data\.installed
)

echo == Activating Virtual Environment ==
if exist mnemonicai_venv\Scripts\activate.bat (
  call mnemonicai_venv\Scripts\activate.bat
) else (
  echo [!] mnemonicai_venv not found. Did install.py finish successfully?
)

echo == Starting MnemonicAi ==
python start.py
pause