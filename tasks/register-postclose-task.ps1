[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $projectRoot 'tasks\run-postclose-research.ps1'
$taskPath = '\DSH-股票助手\'
$taskName = '收盘后数据更新'

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "收盘后更新任务脚本不存在: $runner"
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $projectRoot

# 周一至周五 17:30：收盘且主要数据源已发布当日 K 线之后；
# 错过触发（关机/休眠）由 StartWhenAvailable 在下次开机补跑。
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 17:30

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskPath $taskPath `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description '交易日收盘后把落后最新交易日的标的强制补抓并重跑研究（continuous_research --action postclose）。' `
    -Force | Out-Null

Write-Output "已注册任务: $taskPath$taskName"
Write-Output "触发时间: 周一至周五 17:30（错过自动补跑；补抓+研究最长 4 小时）。"
Write-Output "全天数据已最新时自动跳过重研究，不会空耗。"
