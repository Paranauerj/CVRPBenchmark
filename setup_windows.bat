@echo off
setlocal

echo 📦 Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo 📦 Checking for Scoop...
where scoop >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Scoop not found.
    echo Please install Scoop first:
    echo Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    echo iwr -useb get.scoop.sh ^| iex
    echo Visit https://scoop.sh/ for more info.
    pause
    exit /b 1
)

echo 🛠️ Installing build dependencies via Scoop...
call scoop install mingw cmake make

echo 🐍 Setting up Python environment...
if not exist venv (
    python -m venv venv
    echo ✅ Virtual environment created.
)

echo 📥 Installing Python dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo ------------------------------------------------
echo To start the application, run:
echo   venv\Scripts\activate.bat
echo   streamlit run app.py
echo ------------------------------------------------
pause
