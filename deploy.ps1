# QuantPulse - Full Cloud Deployment to Supabase
# Run: powershell -ExecutionPolicy Bypass -File deploy.ps1
#
# Result after running:
#   Dashboard URL (always online):
#   https://sofowpweliticltlbxrj.supabase.co/storage/v1/object/public/site/index.html

$ErrorActionPreference = "Stop"
$PROJECT_REF = "sofowpweliticltlbxrj"
$SCRIPT_DIR  = $PSScriptRoot
$ENV_FILE    = Join-Path $SCRIPT_DIR ".env"
$HTML_FILE   = Join-Path $SCRIPT_DIR "dashboard_app\index.html"

Write-Host ""
Write-Host "  QuantPulse Cloud Deployment" -ForegroundColor Cyan
Write-Host "  ===========================" -ForegroundColor Cyan
Write-Host ""

# Load .env
if (-not (Test-Path $ENV_FILE)) {
    Write-Host "ERROR: .env not found at $ENV_FILE" -ForegroundColor Red; exit 1
}
$envVars = @{}
Get-Content $ENV_FILE | ForEach-Object {
    if ($_ -match "^\s*([^#][^=\s]+)\s*=\s*(.+)\s*$") { $envVars[$matches[1]] = $matches[2].Trim() }
}
$ALPACA_KEY     = $envVars["ALPACA_API_KEY"]
$ALPACA_SECRET  = $envVars["ALPACA_SECRET_KEY"]
$GITHUB_TOKEN   = $envVars["GITHUB_TOKEN"]
$GITHUB_REPO    = if ($envVars["GITHUB_REPOSITORY"]) { $envVars["GITHUB_REPOSITORY"] } else { "owandakin-coder/agent-analyst" }
$SUPABASE_TOKEN = $envVars["SUPABASE_ACCESS_TOKEN"]
$BROKER_KEY     = $envVars["ATZMA_BROKER_CREDENTIAL_KEY"]

if (-not $ALPACA_KEY -or -not $ALPACA_SECRET) {
    Write-Host "ERROR: Missing Alpaca keys in .env" -ForegroundColor Red; exit 1
}
if (-not $BROKER_KEY) {
    $BROKER_KEY = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
    Write-Host "  Generated ATZMA_BROKER_CREDENTIAL_KEY for this deploy." -ForegroundColor Yellow
}
Write-Host "  .env OK (key: $($ALPACA_KEY.Substring(0,8))...)" -ForegroundColor Green

# Step 1: Auth
Write-Host ""
if ($SUPABASE_TOKEN -and $SUPABASE_TOKEN -notlike "sbp_xxx*") {
    Write-Host "  [1/5] Using SUPABASE_ACCESS_TOKEN from .env (no browser needed)..." -ForegroundColor Green
    $env:SUPABASE_ACCESS_TOKEN = $SUPABASE_TOKEN
} else {
    Write-Host "  [1/5] No SUPABASE_ACCESS_TOKEN in .env - opening browser login..." -ForegroundColor Yellow
    Write-Host "        (Tip: add SUPABASE_ACCESS_TOKEN=sbp_... to .env to skip this step)" -ForegroundColor DarkGray
    npx --yes supabase@latest login
    if ($LASTEXITCODE -ne 0) { Write-Host "Login failed" -ForegroundColor Red; exit 1 }
}

# Step 2: Link project
Write-Host ""
Write-Host "  [2/5] Linking project $PROJECT_REF ..." -ForegroundColor Yellow
Set-Location $SCRIPT_DIR
npx supabase@latest link --project-ref $PROJECT_REF
if ($LASTEXITCODE -ne 0) { Write-Host "Link failed" -ForegroundColor Red; exit 1 }

# Step 3: Push secrets
Write-Host ""
Write-Host "  [3/5] Uploading API secrets (keys stay server-side only)..." -ForegroundColor Yellow
if ($GITHUB_TOKEN) {
    npx supabase@latest secrets set ALPACA_API_KEY=$ALPACA_KEY ALPACA_SECRET_KEY=$ALPACA_SECRET GITHUB_TOKEN=$GITHUB_TOKEN GITHUB_REPOSITORY=$GITHUB_REPO ATZMA_BROKER_CREDENTIAL_KEY=$BROKER_KEY
} else {
    npx supabase@latest secrets set ALPACA_API_KEY=$ALPACA_KEY ALPACA_SECRET_KEY=$ALPACA_SECRET GITHUB_REPOSITORY=$GITHUB_REPO ATZMA_BROKER_CREDENTIAL_KEY=$BROKER_KEY
}
if ($LASTEXITCODE -ne 0) { Write-Host "Secrets failed" -ForegroundColor Red; exit 1 }
Write-Host "  Secrets OK" -ForegroundColor Green

# Step 4: Deploy Edge Function
Write-Host ""
Write-Host "  [4/5] Deploying Edge Function 'api' (Alpaca proxy)..." -ForegroundColor Yellow
npx supabase@latest functions deploy api --no-verify-jwt
if ($LASTEXITCODE -ne 0) { Write-Host "Function deploy failed" -ForegroundColor Red; exit 1 }
Write-Host "  Edge Function OK" -ForegroundColor Green

# Step 5: Upload HTML to Storage
Write-Host ""
Write-Host "  [5/5] Creating public Storage bucket and uploading dashboard..." -ForegroundColor Yellow

npx supabase@latest storage create-bucket site --public 2>$null
Start-Sleep -Seconds 1

npx supabase@latest storage cp --experimental $HTML_FILE ss://site/index.html
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Storage upload failed. Upload manually:" -ForegroundColor Red
    Write-Host "  https://supabase.com/dashboard/project/$PROJECT_REF/storage/buckets" -ForegroundColor Yellow
    Write-Host "  Create bucket 'site' (public) and upload dashboard_app\index.html" -ForegroundColor Yellow
} else {
    Write-Host "  Storage OK" -ForegroundColor Green
}

# Done
Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Cyan
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "  ==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  LIVE Dashboard (share with anyone):" -ForegroundColor White
Write-Host ""
Write-Host "  https://$PROJECT_REF.supabase.co/storage/v1/object/public/site/index.html" -ForegroundColor Green
Write-Host ""
Write-Host "  API always online at:" -ForegroundColor White
Write-Host "  https://$PROJECT_REF.supabase.co/functions/v1/api/account" -ForegroundColor Cyan
Write-Host ""
