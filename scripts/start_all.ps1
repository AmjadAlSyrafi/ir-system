# start_all.ps1 — Start all IR System services on Windows
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1

$Root = Split-Path -Parent $PSScriptRoot

$Services = @(
    @{ Name = "preprocessing";    Port = 8001; Dir = "services\preprocessing" },
    @{ Name = "indexing";         Port = 8002; Dir = "services\indexing" },
    @{ Name = "retrieval";        Port = 8003; Dir = "services\retrieval" },
    @{ Name = "query_refinement"; Port = 8005; Dir = "services\query_refinement" },
    @{ Name = "evaluation";       Port = 8006; Dir = "services\evaluation" },
    @{ Name = "api_gateway";      Port = 8000; Dir = "services\api_gateway" }
)

function Write-OK  ($msg) { Write-Host "  [OK] $msg"      -ForegroundColor Green }
function Write-Err ($msg) { Write-Host "  [FAIL] $msg"    -ForegroundColor Red }
function Write-Info($msg) { Write-Host "  [...] $msg"     -ForegroundColor Yellow }
function Write-Hdr ($msg) { Write-Host "`n$msg"           -ForegroundColor Cyan }

Write-Hdr "============================================"
Write-Hdr "   IR SYSTEM — Starting All Services"
Write-Hdr "============================================"

# ── Start each service in a new minimized window ──────────────────────────────
foreach ($svc in $Services) {
    $svcDir  = Join-Path $Root $svc.Dir
    $logFile = Join-Path $Root "logs\$($svc.Name).log"

    # Ensure logs directory exists
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

    Write-Info "Starting $($svc.Name) on port $($svc.Port)..."

    $cmd = "python -m uvicorn main:app --host 0.0.0.0 --port $($svc.Port) --reload"

    Start-Process -FilePath "powershell" `
        -ArgumentList "-NoExit", "-Command", "cd '$svcDir'; $cmd *> '$logFile'" `
        -WindowStyle Minimized

    Start-Sleep -Seconds 1
}

Write-Hdr "`nWaiting for all services to become healthy..."
Write-Host ""

# ── Wait for each health endpoint ─────────────────────────────────────────────
foreach ($svc in $Services) {
    $url         = "http://localhost:$($svc.Port)/health"
    $maxAttempts = 30
    $attempt     = 0
    $healthy     = $false

    Write-Host "  Waiting for $($svc.Name) ($url)..." -NoNewline

    while ($attempt -lt $maxAttempts) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        } catch { }
        $attempt++
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline
    }

    if ($healthy) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " TIMEOUT" -ForegroundColor Red
        Write-Err "$($svc.Name) did not become healthy after $maxAttempts seconds."
        Write-Err "Check logs\$($svc.Name).log for errors."
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Hdr "`n============================================"
Write-Hdr "   All Services Running"
Write-Hdr "============================================"
Write-Host ""
Write-Host "  API Gateway:       http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:          http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Preprocessing:     http://localhost:8001/docs" -ForegroundColor White
Write-Host "  Indexing:          http://localhost:8002/docs" -ForegroundColor White
Write-Host "  Retrieval:         http://localhost:8003/docs" -ForegroundColor White
Write-Host "  Query Refinement:  http://localhost:8005/docs" -ForegroundColor White
Write-Host "  Evaluation:        http://localhost:8006/docs" -ForegroundColor White
Write-Host ""
Write-Host "  Logs folder:  $Root\logs\" -ForegroundColor Gray
Write-Host ""
Write-Host "  To stop all services: close the minimized windows" -ForegroundColor Gray
Write-Host "  Or run: scripts\stop_all.ps1" -ForegroundColor Gray
Write-Host ""