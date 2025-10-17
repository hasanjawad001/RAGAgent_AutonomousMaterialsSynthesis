@echo off
setlocal
set "UV_NATIVE_TLS=true"
set UV_LINK_MODE=copy

REM === Check if uv is installed ===
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] uv not found. Installing uv via PowerShell...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex; $env:Path += ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"
    echo.
    REM === Verify uv installation ===
    where uv >nul 2>nul
    if errorlevel 1 (
        echo [⚠️] uv installed but not detected in PATH.
        echo [⚠️] Please CLOSE and REOPEN this window or open a new terminal before re-running this script.
        echo [⚠️] Press any key to exit...
        pause >nul
        exit /b
    )
)

REM === Set up virtual environment ===
if not exist .venv (
    echo [INFO] Creating virtual environment with uv...
    uv venv --python 3.12
)

REM === Activate virtual environment ===
call .venv\Scripts\activate.bat

REM === Install dependencies ===
echo [INFO] Installing dependencies...
uv pip install -r requirements.txt

REM === Fix for PATH in some cases ===
call .venv\Scripts\activate.bat

REM === Launch the app ===
echo [INFO] Launching Streamlit app...
uv run streamlit run app.py --server.port=8501

pause
endlocal
