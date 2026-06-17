# stop_all.ps1 — Kill all IR System uvicorn processes on Windows
# Usage: powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1

$ports = @(8000, 8001, 8002, 8003, 8005, 8006)

Write-Host "`nStopping IR System services..." -ForegroundColor Cyan
Write-Host ""

foreach ($port in $ports) {
    $process = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
               Select-Object -First 1

    if ($process) {
        $pid = $process.OwningProcess
        $name = (Get-Process -Id $pid -ErrorAction SilentlyContinue).Name
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Write-Host "  [OK] Stopped port $port  (PID $pid  $name)" -ForegroundColor Green
    } else {
        Write-Host "  [--] Port $port not in use" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "All services stopped." -ForegroundColor Cyan
Write-Host ""