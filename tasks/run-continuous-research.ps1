[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$projectRoot = Split-Path -Parent $PSScriptRoot
$quantRoot = Join-Path $projectRoot 'quant'
$python = Join-Path $quantRoot '.venv\Scripts\python.exe'
$entry = Join-Path $quantRoot 'scripts\continuous_research.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "独立 Python 环境不存在: $python"
}
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "连续研究调度器不存在: $entry"
}

& $python $entry --action once
$exitCode = $LASTEXITCODE
exit $exitCode
