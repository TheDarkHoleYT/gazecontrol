@echo off
REM ============================================================
REM  GazeControl — Launcher with mode selection
REM ============================================================
REM  Activates .venv and starts the app.
REM  Lets the user pick the input mode (hand-only / eye+hand)
REM  or fall back to the in-app Qt selector dialog.
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\gazecontrol.exe" (
    echo [ERROR] .venv not found or gazecontrol not installed.
    echo Run install.bat first.
    pause
    exit /b 1
)

set "GC=.venv\Scripts\gazecontrol.exe"

echo.
echo ============================================================
echo                       GazeControl
echo ============================================================
echo.
echo   1. Hand Only          ^(pinch / drag / scroll^)
echo   2. Eye + Hand         ^(gaze target + hand action^)
echo   3. Show Qt selector   ^(in-app dialog^)
echo   4. Calibrate gaze     ^(3x3 grid, eye+hand only^)
echo   5. Doctor             ^(probe camera / models / deps^)
echo   6. Quit
echo.

choice /C 123456 /N /M "Choose [1-6]: "
set "PICK=%ERRORLEVEL%"

if "%PICK%"=="1" (
    "%GC%" --mode hand --no-mode-selector %*
    goto :end
)
if "%PICK%"=="2" (
    "%GC%" --mode eye-hand --no-mode-selector %*
    goto :end
)
if "%PICK%"=="3" (
    "%GC%" %*
    goto :end
)
if "%PICK%"=="4" (
    "%GC%" --calibrate-gaze
    goto :end
)
if "%PICK%"=="5" (
    "%GC%" --doctor
    pause
    goto :end
)
if "%PICK%"=="6" (
    goto :end
)

:end
endlocal
exit /b 0
