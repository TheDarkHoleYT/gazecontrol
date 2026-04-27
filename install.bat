@echo off
REM ============================================================
REM  GazeControl — Fast Installer (Windows)
REM ============================================================
REM  Steps:
REM   1. Find Python 3.11+ (py launcher fallback to python).
REM   2. Create .venv if missing.
REM   3. Upgrade pip + install gazecontrol in editable mode with [eye] extra.
REM   4. Optionally download the L2CS-Net ONNX model.
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo === GazeControl installer ===
echo Working directory: %CD%
echo.

REM ---------- 1. Locate Python ----------
set "PY="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3.11 -V >nul 2>&1 && set "PY=py -3.11"
    if not defined PY ( py -3.12 -V >nul 2>&1 && set "PY=py -3.12" )
    if not defined PY ( py -3 -V >nul 2>&1 && set "PY=py -3" )
)
if not defined PY (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=python"
)
if not defined PY (
    echo [ERROR] No Python interpreter found. Install Python 3.11+ from python.org.
    pause
    exit /b 1
)
echo Using Python: %PY%
%PY% -V

REM ---------- 2. Create venv ----------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv || ( echo [ERROR] venv creation failed. & pause & exit /b 2 )
) else (
    echo Virtual environment .venv already exists.
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"

REM ---------- 3. Ensure pip is present ----------
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo pip missing in .venv — bootstrapping with ensurepip ...
    "%VENV_PY%" -m ensurepip --upgrade
    "%VENV_PY%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo ensurepip failed — recreating .venv from scratch ...
        rmdir /s /q .venv
        %PY% -m venv .venv || ( echo [ERROR] venv recreate failed. & pause & exit /b 2 )
        "%VENV_PY%" -m ensurepip --upgrade || ( echo [ERROR] pip bootstrap failed. & pause & exit /b 3 )
    )
)

REM ---------- 4. Upgrade pip + install package ----------
echo.
echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools || ( echo [ERROR] pip upgrade failed. & pause & exit /b 3 )

echo.
echo Installing gazecontrol[eye] in editable mode ...
"%VENV_PY%" -m pip install -e ".[eye]" || ( echo [ERROR] install failed. & pause & exit /b 4 )

REM ---------- 5. Optional model download ----------
echo.
echo NOTE: L2CS download also needs torch + torchvision + onnx (~2 GB).
choice /C YN /N /M "Download L2CS-Net ONNX model now (~100 MB + ~2 GB deps)? [Y/N] "
if errorlevel 2 goto :skip_model
if errorlevel 1 (
    if not exist "tools\download_l2cs.py" (
        echo [WARN] tools\download_l2cs.py not found; skipping.
        goto :skip_model
    )
    echo Installing torch + torchvision + onnx + gdown ...
    "%VENV_PY%" -m pip install torch torchvision onnx gdown || (
        echo [WARN] torch/onnx install failed; cannot convert L2CS model.
        goto :skip_model
    )
    echo Downloading + converting L2CS-Net model ...
    "%VENV_PY%" "tools\download_l2cs.py" || echo [WARN] Model download failed; you can re-run later.

    REM torch may upgrade numpy past 2.x — eyetrax pins numpy<2.
    echo.
    echo Pinning numpy^<2 for eyetrax compatibility ...
    "%VENV_PY%" -m pip install "numpy<2" --force-reinstall --no-deps || echo [WARN] numpy pin failed.
)
:skip_model

REM ---------- 6. Done ----------
echo.
echo === Install complete ===
echo Next steps:
echo   run.bat                   ^(launches GazeControl with mode chooser^)
echo   .venv\Scripts\gazecontrol --doctor
echo.
endlocal
pause
exit /b 0
