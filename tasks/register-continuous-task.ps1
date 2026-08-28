[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $projectRoot 'tasks\run-continuous-research.ps1'
$taskPath = '\DSH-股票助手\'
$taskName = '全市场连续研究'
$oldTaskName = '每日研究'

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "连续研究任务脚本不存在: $runner"
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $projectRoot

$firstRun = (Get-Date).AddMinutes(1)
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $firstRun `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

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
    -Description 'DSH 全市场股票、场内 ETF、场外基金数据持续获取与失败自动续跑。' `
    -Force | Out-Null

$oldTask = Get-ScheduledTask -TaskPath $taskPath -TaskName $oldTaskName -ErrorAction SilentlyContinue
if ($null -ne $oldTask) {
    Disable-ScheduledTask -TaskPath $taskPath -TaskName $oldTaskName | Out-Null
    Write-Output "已禁用旧任务（未删除）: $taskPath$oldTaskName"
}

Write-Output "已注册任务: $taskPath$taskName"
Write-Output "首次触发时间: $($firstRun.ToString('o'))，之后每 10 分钟触发。"
