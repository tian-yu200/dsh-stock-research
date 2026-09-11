[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $projectRoot 'tasks\run-fullscan-guardian.ps1'
$taskPath = '\DSH-股票助手\'
$taskName = '全市场扫描监护'

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "扫描监护任务脚本不存在: $runner"
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $projectRoot

# 与“全市场连续研究”（每 10 分钟）错峰：监护每 5 分钟巡检一次，
# 单次 tick 只读有界 JSON 聚合 + 日志尾部（<10s），不启动第二波 worker。
$firstRun = (Get-Date).AddMinutes(1)
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $firstRun `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 2)

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
    -Description 'DSH 全市场扫描监护：只读巡检 latest_status/manifest/rule_gates；瞬时暂停调用调度器 resume() 恢复；确定性门槛失败只记录不修；manifest+current.json 双 success 即写 done 旗并自注销，任务结束。' `
    -Force | Out-Null

Write-Output "已注册任务: $taskPath$taskName"
Write-Output "首次触发时间: $($firstRun.ToString('o'))，之后每 5 分钟巡检。"
Write-Output "结束条件: manifest.json 与 current.json 同 run_id 双 success（结果已发布）。"
