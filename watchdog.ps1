# watchdog.ps1
# ============
# Monitors the agent process and restarts it if it crashes.
# Runs at Windows login via Task Scheduler (no admin needed).
# Logs to watchdog.log in the project directory.

$ProjectDir  = "C:\Users\Ea Arage\Downloads\agent analyst"
$PythonExe   = "python"
$AgentArgs   = "main.py --mode live_paper --auto-approve"
$LogFile     = Join-Path $ProjectDir "agent_service.log"
$WatchdogLog = Join-Path $ProjectDir "watchdog.log"
$CheckSec    = 60   # check every 60 seconds

function Write-WLog($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [Watchdog] $msg"
    Write-Host $line
    Add-Content -Path $WatchdogLog -Value $line -Encoding UTF8
}

Write-WLog "Watchdog started. Monitoring agent every ${CheckSec}s ..."

while ($true) {
    # Check if the agent Python process is running
    $agentProc = Get-Process -Name python -ErrorAction SilentlyContinue |
                 Where-Object {
                     try { $_.MainModule.FileName -ne $null } catch { $false }
                 }

    if (-not $agentProc) {
        Write-WLog "Agent not running. Starting ..."
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName               = $PythonExe
            $psi.Arguments              = $AgentArgs
            $psi.WorkingDirectory       = $ProjectDir
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError  = $true
            $psi.UseShellExecute        = $false
            $psi.CreateNoWindow         = $true

            $proc = New-Object System.Diagnostics.Process
            $proc.StartInfo = $psi

            # Async log stdout/stderr to agent_service.log
            $logStream = [System.IO.StreamWriter]::new($LogFile, $true, [System.Text.Encoding]::UTF8)
            $logStream.AutoFlush = $true

            $proc.add_OutputDataReceived({
                param($s, $e)
                if ($e.Data) { $logStream.WriteLine($e.Data) }
            })
            $proc.add_ErrorDataReceived({
                param($s, $e)
                if ($e.Data) { $logStream.WriteLine("ERR: " + $e.Data) }
            })

            $proc.Start()         | Out-Null
            $proc.BeginOutputReadLine()
            $proc.BeginErrorReadLine()

            Write-WLog "Agent started. PID=$($proc.Id)"
        } catch {
            Write-WLog "ERROR starting agent: $_"
        }
    }

    Start-Sleep -Seconds $CheckSec
}
