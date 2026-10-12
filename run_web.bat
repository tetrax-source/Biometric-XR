@echo off
title Biometric Attendance Web Portal
cd /d "%~dp0"
echo ========================================================
echo  Starting Biometric Attendance Web Portal on localhost...
echo ========================================================
echo Opening browser in 3 seconds...
start "" "http://localhost:5000"
python app_web.py
pause
