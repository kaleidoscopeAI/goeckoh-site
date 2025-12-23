@echo off
:: --- KALEIDOSCOPE AI HATCHING SCRIPT (WINDOWS) ---

echo Initializing AI Egg... Please wait.

:: Get the directory of the batch file
set SCRIPT_DIR=%~dp0
set BACKEND_DIR=%SCRIPT_DIR%system\backend
set FRONTEND_DIR=%SCRIPT_DIR%system\frontend\build
set PYTHON_EXEC=%SCRIPT_DIR%system\portable_runtime\python_win\python.exe

:: --- Step 1: Launch AI Backend Server ---
echo [1/3] Awakening cognitive core (Backend)...
cd /d "%BACKEND_DIR%"
start "AI Backend" /b cmd /c "venv\Scripts\activate.bat && %PYTHON_EXEC% app.py"
echo Backend process started.
timeout /t 5 >nul

:: --- Step 2: Launch Visualization Server ---
echo [2/3] Igniting visualization layer (Frontend)...
cd /d "%FRONTEND_DIR%"
start "AI Frontend" /b cmd /c "%PYTHON_EXEC% -m http.server 8000"
echo Frontend server started.
timeout /t 2 >nul

:: --- Step 3: Open Browser and Hatch AI ---
echo [3/3] Hatching complete. Opening interface in your browser...
start http://localhost:8000

echo.
echo ✅ The AI is now live. Close the two new console windows to shut down the system.
pause
