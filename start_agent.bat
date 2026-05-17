@echo off
cd /d "C:\Users\Ea Arage\Downloads\agent analyst"
echo [%date% %time%] Agent starting... >> agent_service.log
python main.py --mode live_paper --auto-approve >> agent_service.log 2>&1
echo [%date% %time%] Agent exited with code %ERRORLEVEL% >> agent_service.log
