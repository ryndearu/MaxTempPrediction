@echo off
echo =====================================
echo   Prediksi Suhu Semarang - Streamlit
echo =====================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python tidak ditemukan! Pastikan Python terinstall dan ada di PATH.
    pause
    exit /b 1
)

echo Python terdeteksi.
echo.

echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting Streamlit application...
echo.
echo =====================================
echo   Aplikasi akan terbuka di browser
echo   Tekan Ctrl+C untuk menghentikan
echo =====================================
echo.

streamlit run streamlit_app.py

pause
