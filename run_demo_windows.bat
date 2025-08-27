@echo off
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
REM pip install --upgrade pip
pip install -r requirements.txt

REM Fix environment not recognizing newly installed streamlit
call venv\Scripts\activate.bat

streamlit run app.py --server.port=8501
pause
