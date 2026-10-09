[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DshArguments
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dshHome = Join-Path $projectRoot '.dsh'
$patch = Join-Path $dshHome 'cordis.patch.yml'

# DSH 安装位置因机器而异：优先环境变量 DSH_CMD，其次 PATH 中的 dsh，最后回退到本地默认安装路径。
$dsh = $env:DSH_CMD
if (-not $dsh) {
    $onPath = Get-Command 'dsh' -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath) {
        $dsh = $onPath.Source
    } elseif (Test-Path -LiteralPath 'D:\deepseek-harness\node_modules\.bin\dsh.cmd' -PathType Leaf) {
        $dsh = 'D:\deepseek-harness\node_modules\.bin\dsh.cmd'
    }
}

if (-not $dsh) {
    throw 'DSH 运行时未找到：请设置环境变量 DSH_CMD 指向 dsh.cmd，或把 dsh 加入 PATH。'
}
if (-not (Test-Path -LiteralPath $dsh -PathType Leaf)) {
    throw "DSH 运行时不存在: $dsh"
}
if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) {
    throw "股票助手 DSH patch 不存在: $patch"
}

$hasPortArgument = $DshArguments | Where-Object {
    $_ -eq '--port' -or $_ -like '--port=*'
}
if (-not $hasPortArgument) {
    $DshArguments = @('--port', '58565') + $DshArguments
}

$env:DSH_HOME = $dshHome
$env:DSH_CWD = $projectRoot
Set-Location -LiteralPath $projectRoot

& $dsh --profile web --patch $patch @DshArguments
exit $LASTEXITCODE
