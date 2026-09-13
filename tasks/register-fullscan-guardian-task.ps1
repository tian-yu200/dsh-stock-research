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
    -Description 'DSH 全市场扫描监护：只读巡检 latest_status/manifest/rule_gates；无在跑作业时保持被动(idle，非交易日的 paused 不触发 resume)；瞬时暂停调用调度器 resume() 恢复；确定性门槛失败只记录不修；manifest 与 current.json 同为已发布态(success/partial)且 done 旗 job_id 匹配当前作业时才写旗并自注销。done 旗为全局单文件，跨轮残留不生效。' `
    -Force | Out-Null

Write-Output "已注册任务: $taskPath$taskName"
Write-Output "首次触发时间: $($firstRun.ToString('o'))，之后每 5 分钟巡检。"
Write-Output "结束条件: manifest.json 与 current.json 同 run_id 且同为已发布态（success/partial）。"
Write-Output "注意: done 旗（guardian-done.json）按 job_id 比对，上一轮残留不会让本轮监护空转；重新注册前无需手动删除，但仍建议清理。"
Write-Output "无在跑作业时监护保持 idle，不会调用 resume()。"
