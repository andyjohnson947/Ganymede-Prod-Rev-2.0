@echo off
REM Regime HMM Auto-Retrain Scheduler
REM Checks model age and retrains if older than 3 days
REM Run via Windows Task Scheduler every 12 hours
REM
REM Task Scheduler setup:
REM   schtasks /create /tn "Ganymede_Regime_Retrain" /tr "C:\Users\Administrator\.claude-worktrees\Ganymede-Prod-Rev-2.0\funny-meninsky\ml_system\regime_retrain.bat" /sc DAILY /st 04:00 /ri 720 /du 24:00 /f
REM   (Runs every 12 hours starting at 04:00 UTC)

cd /d "C:\Users\Administrator\.claude-worktrees\Ganymede-Prod-Rev-2.0\funny-meninsky"

echo [%date% %time%] Regime auto-retrain starting... >> ml_system\outputs\regime_retrain.log

"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" ml_system\regime_hmm.py --auto-retrain >> ml_system\outputs\regime_retrain.log 2>&1

echo [%date% %time%] Regime auto-retrain complete. >> ml_system\outputs\regime_retrain.log
echo. >> ml_system\outputs\regime_retrain.log
