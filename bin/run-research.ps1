[CmdletBinding()]
param(
    [ValidateSet('daily', 'monthly', 'on_demand')]
    [string]$Mode = 'on_demand',

    [ValidateSet('all', 'stock', 'etf', 'otc_fund')]
    [string]$AssetScope = 'all',

    [ValidateSet('quick', 'standard', 'full')]
    [string]$Depth,

    [string[]]$Symbols
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'quant\.venv\Scripts\python.exe'
$entry = Join-Path $projectRoot 'quant\scripts\run_assistant_research.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "独立 Python 环境不存在: $python"
}
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "研究入口不存在: $entry"
}

$arguments = @($entry, '--mode', $Mode, '--asset-scope', $AssetScope)
if ($Depth) {
    $arguments += @('--depth', $Depth)
}
if ($null -ne $Symbols -and $Symbols.Count -gt 0) {
    if ($Depth -eq 'full') {
        throw 'depth=full 会刷新整个代码池，不能与 -Symbols 同时使用；请用 quick 或 standard。'
    }
    $arguments += '--symbols'
    $arguments += $Symbols
}

& $python @arguments
$exitCode = $LASTEXITCODE
exit $exitCode
