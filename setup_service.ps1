# setup_service.ps1
# מגדיר את Agent Analyst כמשימה מתוזמנת שמתחילה עם Windows

$ProjectDir = "C:\Users\Ea Arage\Downloads\agent analyst"
$PythonExe  = (Get-Command python).Source
$TaskName   = "AgentAnalyst"
$LogFile    = "$ProjectDir\agent_service.log"

# הסר משימה קיימת אם יש
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# הגדר את הפעולה
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "main.py --mode live_paper --auto-approve" `
    -WorkingDirectory $ProjectDir

# הפעל בעת אתחול המחשב
$Trigger = New-ScheduledTaskTrigger -AtStartup

# הגדרות: רץ גם כשלא מחובר, מתחיל מחדש אם נכשל
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# רשום את המשימה
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -RunLevel Highest `
    -Force | Out-Null

# הפעל מיד
Start-ScheduledTask -TaskName $TaskName

Write-Host "AgentAnalyst service registered and started."
Write-Host "To check status: Get-ScheduledTask -TaskName AgentAnalyst"
Write-Host "To stop:         Stop-ScheduledTask -TaskName AgentAnalyst"
Write-Host "To remove:       Unregister-ScheduledTask -TaskName AgentAnalyst -Confirm:`$false"
