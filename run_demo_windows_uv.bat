@echo off
set UV_LINK_MODE=copy

REM === Check if uv is installed ===
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] uv not found. Installing pipx and uv...
    pip install pipx
    pipx ensurepath
    echo [INFO] Now installing uv with pipx...
    pipx install uv
    echo.
    echo [⚠️] Please CLOSE and REOPEN this window or open a new terminal before re-running this script.
    echo [⚠️] Press any key to exit...
    pause >nul
    exit /b
)

REM === Set up virtual environment ===
if not exist .venv (
    uv venv
)
call venv\Scripts\activate.bat

REM === Install dependencies ===
uv pip install -r requirements.txt

REM === Fix for PATH in some cases ===
call venv\Scripts\activate.bat

REM === Launch the app ===
uv run streamlit run app.py --server.port=8501

pause