$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$logsDir = Join-Path $root "logs"
$guardLog = Join-Path $logsDir "realtime_guard.log"
$loopLog = Join-Path $logsDir "realtime_loop.log"
$loopErr = Join-Path $logsDir "realtime_loop.err"
$bootstrapLog = Join-Path $logsDir "realtime_guard.bootstrap.log"
$pythonExe = "C:\Users\Ea Arage\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if (-not (Test-Path $pythonExe)) {
  $pythonExe = (Get-Command python).Source
}

Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)\s*$') {
    $name = $matches[1].Trim()
    $value = $matches[2].Trim().Trim('"')
    Set-Item -Path ("Env:" + $name) -Value $value
  }
}

if (-not $env:ATZMA_FAIL_CLOSED_CONTROL) { $env:ATZMA_FAIL_CLOSED_CONTROL = "1" }
if (-not $env:ATZMA_REQUIRE_FRESH_QUOTES) { $env:ATZMA_REQUIRE_FRESH_QUOTES = "1" }
if (-not $env:ATZMA_MAX_QUOTE_AGE_SECONDS) { $env:ATZMA_MAX_QUOTE_AGE_SECONDS = "120" }
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
Add-Content -Path $bootstrapLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') bootstrap ok"

while ($true) {
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -Path $guardLog -Value "$stamp START main.py --mode live_paper --auto-approve"
  try {
    $proc = Start-Process -FilePath $pythonExe `
      -ArgumentList @("main.py", "--mode", "live_paper", "--auto-approve") `
      -WorkingDirectory $root `
      -NoNewWindow `
      -RedirectStandardOutput $loopLog `
      -RedirectStandardError $loopErr `
      -PassThru `
      -Wait
    Add-Content -Path $guardLog -Value "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) EXIT CODE $($proc.ExitCode)"
  } catch {
    Add-Content -Path $guardLog -Value "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) PROCESS ERROR $($_.Exception.Message)"
  }
  $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -Path $guardLog -Value "$endStamp RESTART in 15s"
  Start-Sleep -Seconds 15
}
