[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$projectRoot = Split-Path -Parent $PSScriptRoot
$quantRoot = Join-Path $projectRoot 'quant'
$python = Join-Path $quantRoot '.venv\Scripts\python.exe'
$entry = Join-Path $quantRoot 'scripts\fullscan_guardian.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "独立 Python 环境不存在: $python"
}
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "扫描监护脚本不存在: $entry"
}

& $python $entry --action tick
$exitCode = $LASTEXITCODE
if ($exitCode -eq 10) {
    Write-Output "全市场扫描已完成且结果已发布，监护任务逻辑结束（后续 tick 直接退出）。"
    exit 0
}
exit $exitCode
