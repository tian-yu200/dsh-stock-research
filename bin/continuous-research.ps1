[CmdletBinding()]
param(
    [ValidateSet('status', 'start', 'pause', 'resume', 'errors', 'coverage')]
    [string]$Action = 'status',
    [int]$Limit = 50,
    [string]$Reason = 'paused by operator'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'quant\.venv\Scripts\python.exe'
$entry = Join-Path $projectRoot 'quant\scripts\continuous_research.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "独立 Python 环境不存在: $python"
}
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "连续研究调度器不存在: $entry"
}

$arguments = @($entry, '--action', $Action)
if ($Action -eq 'errors') {
    $arguments += @('--limit', $Limit)
}
if ($Action -eq 'pause') {
    $arguments += @('--reason', $Reason)
}

& $python @arguments
$exitCode = $LASTEXITCODE
exit $exitCode
