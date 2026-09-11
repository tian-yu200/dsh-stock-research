[CmdletBinding()]
param(
    [switch]$RestoreDailyTask
)

$ErrorActionPreference = 'Stop'
$taskPath = '\DSH-股票助手\'
$taskName = '全市场连续研究'

Unregister-ScheduledTask `
    -TaskPath $taskPath `
    -TaskName $taskName `
    -Confirm:$false `
    -ErrorAction SilentlyContinue

if ($RestoreDailyTask) {
    Enable-ScheduledTask `
        -TaskPath $taskPath `
        -TaskName '每日研究' `
        -ErrorAction SilentlyContinue | Out-Null
}

Write-Output "已注销任务（若存在）: $taskPath$taskName"
