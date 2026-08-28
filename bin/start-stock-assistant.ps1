[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DshArguments
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dshHome = Join-Path $projectRoot '.dsh'
$patch = Join-Path $dshHome 'cordis.patch.yml'
$dsh = 'D:\deepseek-harness\node_modules\.bin\dsh.cmd'

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
