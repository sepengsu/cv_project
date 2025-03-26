@echo off
REM Create a virtual environment with Python 3.11
virtualenv -p python3.11 venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip to the latest version
python -m pip install --upgrade pip

REM Install the required packages
pip install -r setting.txt

echo Virtual environment setup complete.