[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskPath = '\DSH-股票助手\'
$taskName = '收盘后数据更新'

$task = Get-ScheduledTask -TaskPath $taskPath -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Output "任务不存在: $taskPath$taskName"
    exit 0
}
Unregister-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Confirm:$false
Write-Output "已删除任务: $taskPath$taskName"
