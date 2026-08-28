[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskPath = '\DSH-股票助手\'
$taskName = '每日研究'

Unregister-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Output "已注销任务: $taskPath$taskName"
