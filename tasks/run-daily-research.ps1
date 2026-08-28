[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$quantRoot = Join-Path $projectRoot 'quant'
$logs = Join-Path $quantRoot 'logs'
$calendarScript = Join-Path $quantRoot 'scripts\trading_calendar.py'
$runner = Join-Path $projectRoot 'bin\run-research.ps1'
$python = Join-Path $quantRoot '.venv\Scripts\python.exe'
$timeoutSeconds = 1800
$taskName = 'DSH 股票助手研究'
$mutex = $null
$process = $null
$stdoutPath = $null
$stderrPath = $null

New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Add-TaskLog {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date).ToString('o'), $Message
    Add-Content -LiteralPath (Join-Path $logs 'scheduled-research.log') -Value $line -Encoding UTF8
}

function Record-TaskFailure {
    param(
        [ValidateSet('scheduled_task_failed', 'scheduled_task_timeout', 'scheduled_task_start_failed')]
        [string]$Kind,
        [string]$Title,
        [string]$Message
    )
    try {
        if (Test-Path -LiteralPath $python -PathType Leaf) {
            & $python -m assistant.task_runtime --kind $Kind --title $Title --message $Message 2>> (Join-Path $logs 'scheduled-research.stderr.log') | Out-Null
        }
    } catch {
        Add-TaskLog "写入失败通知失败: $($_.Exception.Message)"
    }
}

function Show-StockToast {
    param([string]$Title, [string]$Message)
    try {
        $burntToast = Get-Command New-BurntToastNotification -ErrorAction SilentlyContinue
        if ($null -ne $burntToast) {
            New-BurntToastNotification -Text $Title, $Message | Out-Null
            return
        }
        Add-Type -AssemblyName System.Runtime.WindowsRuntime -ErrorAction SilentlyContinue
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$Title</text><text>$Message</text></binding></visual></toast>")
        $notification = New-Object Windows.UI.Notifications.ToastNotification $xml
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('DSH 股票助手').Show($notification)
    } catch {
        Add-TaskLog "系统通知不可用: $($_.Exception.Message)"
    }
}

try {
    $mutex = New-Object System.Threading.Mutex($false, 'Local\DSH-Stock-Assistant-Research')
    if (-not $mutex.WaitOne(0)) {
        Add-TaskLog "已有研究任务运行，跳过本次调度。"
        exit 0
    }

    $today = (Get-Date).ToString('yyyy-MM-dd')
    if (-not (Test-Path -LiteralPath $calendarScript -PathType Leaf)) {
        throw "交易日历脚本不存在: $calendarScript"
    }
    $calendarOutput = (& $python $calendarScript --date $today `
        2>> (Join-Path $logs 'trading-calendar.stderr.log') | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "交易日历更新失败: $calendarOutput"
    }
    $calendarState = $calendarOutput | ConvertFrom-Json
    Add-TaskLog "交易日历来源=$($calendarState.source)，覆盖=$($calendarState.coverage_start)..$($calendarState.coverage_end)。"
    if (-not $calendarState.is_trading_day) {
        Add-TaskLog "$today 不是交易日，跳过研究。"
        exit 0
    }

    $mode = if ($calendarState.is_month_end_trading_day) { 'monthly' } else { 'daily' }
    Add-TaskLog "开始 $mode 研究，日期=$today。"

    $stdoutPath = Join-Path $logs "scheduled-$today.stdout.log"
    $stderrPath = Join-Path $logs "scheduled-$today.stderr.log"
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    $argumentList = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $runner,
        '-Mode', $mode,
        '-AssetScope', 'all'
    )
    try {
        $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentList `
            -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden
    } catch {
        Record-TaskFailure -Kind scheduled_task_start_failed -Title "$taskName 启动失败" `
            -Message $_.Exception.Message
        Show-StockToast -Title "$taskName 启动失败" -Message $_.Exception.Message
        exit 2
    }

    if (-not $process.WaitForExit($timeoutSeconds * 1000)) {
        Add-TaskLog "研究超过 $timeoutSeconds 秒，终止进程树。"
        & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null
        Record-TaskFailure -Kind scheduled_task_timeout -Title "$taskName 超时" `
            -Message "研究任务超过 $timeoutSeconds 秒，保留上一次成功快照。"
        Show-StockToast -Title "$taskName 超时" -Message '已保留上一次成功快照。'
        exit 124
    }

    $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -Raw -LiteralPath $stdoutPath } else { '' }
    $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -Raw -LiteralPath $stderrPath } else { '' }
    Add-TaskLog "研究进程结束，退出码=$($process.ExitCode)。"
    if ($stderr) {
        Add-TaskLog "研究 stderr 已写入 $stderrPath。"
    }

    if ($process.ExitCode -ne 0) {
        $parsed = $null
        try {
            $parsed = $stdout.Trim() | ConvertFrom-Json
        } catch {
            $parsed = $null
        }
        if ($null -eq $parsed -or $parsed.status -eq 'error') {
            Record-TaskFailure -Kind scheduled_task_failed -Title "$taskName 失败" `
                -Message "研究进程退出码 $($process.ExitCode)。详见 $stdoutPath 和 $stderrPath。"
            Show-StockToast -Title "$taskName 失败" -Message '已保留上一次成功快照，请查看 DSH 收件箱。'
        }
        exit $process.ExitCode
    }

    $parsed = $null
    try {
        $parsed = $stdout.Trim() | ConvertFrom-Json
    } catch {
        $parsed = $null
    }
    if ($null -ne $parsed -and $parsed.status -eq 'queued') {
        Add-TaskLog "全市场研究已入队，job_id=$($parsed.job_id)，后续按批次推进。"
        Show-StockToast -Title "$taskName 已入队" -Message '全市场数据将分批抓取，可通过 get_run_status 查询进度。'
    } else {
        Show-StockToast -Title "$taskName 完成" -Message "$today 的研究结果已写入 DSH 收件箱。"
    }
    exit 0
} catch {
    Add-TaskLog "调度异常: $($_.Exception.Message)"
    Record-TaskFailure -Kind scheduled_task_failed -Title "$taskName 异常" `
        -Message $_.Exception.Message
    Show-StockToast -Title "$taskName 异常" -Message '已保留上一次成功快照，请查看日志。'
    exit 3
} finally {
    if ($null -ne $mutex) {
        try { $mutex.ReleaseMutex() } catch {}
        $mutex.Dispose()
    }
}
