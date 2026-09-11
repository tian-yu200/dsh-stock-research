[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskPath = '\DSH-股票助手\'
$taskName = '全市场扫描监护'

Unregister-ScheduledTask `
    -TaskPath $taskPath `
    -TaskName $taskName `
    -Confirm:$false `
    -ErrorAction SilentlyContinue

Write-Output "已注销任务（若存在）: $taskPath$taskName"
