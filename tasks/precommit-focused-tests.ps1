[CmdletBinding()]
param()

# T17 提交前聚焦测试门禁包装器（2026-10-12）：
#   - 解释器固定取**主检出**的 quant\.venv\Scripts\python.exe（主检出 =
#     git rev-parse --path-format=absolute --git-common-dir 的父目录），
#     使 linked worktree 也能用同一套测试门禁；
#   - 测试文件与 -q -p no:warnings 与 T6 门禁完全一致，不放宽、不跳过；
#   - 在**当前检出**根目录运行（测试读当前分支的代码）；
#   - 当前检出缺 quant/data/benchmarks/trade_calendar.csv 时立即失败并提示，
#     绝不自动拷贝。

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
# hook 输出经管道重定向：统一按 UTF-8 编解码，保证提示与 pytest 输出不乱码。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$currentRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $currentRoot

$gitCommonDir = & git rev-parse --path-format=absolute --git-common-dir
if ($LASTEXITCODE -ne 0 -or -not $gitCommonDir) {
    Write-Host '[precommit] 无法解析 git 公共目录（git rev-parse --path-format=absolute --git-common-dir 失败）'
    exit 1
}
$mainRoot = Split-Path -Parent $gitCommonDir

$python = Join-Path $mainRoot 'quant\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Host "[precommit] 主检出解释器不存在: $python"
    exit 1
}

$calendar = Join-Path $currentRoot 'quant\data\benchmarks\trade_calendar.csv'
if (-not (Test-Path -LiteralPath $calendar -PathType Leaf)) {
    $source = Join-Path $mainRoot 'quant\data\benchmarks\trade_calendar.csv'
    Write-Host "[precommit] 当前检出缺少交易日历: $calendar"
    Write-Host '[precommit] 聚焦测试依赖它。请临时拷入主检出日历后重试（不要自动拷贝，不要提交该文件）：'
    Write-Host "[precommit]   Copy-Item '$source' '$calendar'"
    Write-Host '[precommit] 测完后删除即可。'
    exit 1
}

$testFiles = @(
    'quant/tests/test_market_data.py',
    'quant/tests/test_data_source_breaker.py',
    'quant/tests/test_data_source_ths.py',
    'quant/tests/test_fetch_stock_worker.py',
    'quant/tests/test_unit_guard.py',
    'quant/tests/test_assistant.py',
    'quant/tests/test_research_snapshot.py',
    'quant/tests/test_gate_policy.py',
    'quant/tests/test_backtest_tradability.py'
)

& $python -m pytest @testFiles -q -p no:warnings
exit $LASTEXITCODE
