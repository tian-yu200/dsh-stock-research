param([string]$Tag)
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$qr = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$qr = Join-Path $qr "quant"
$runDir = Join-Path $qr "logs\codex-runs"
$out = Join-Path $runDir "$Tag.out.log"
$exitFile = Join-Path $runDir "$Tag.exitcode"
$py = Join-Path $qr ".venv\Scripts\python.exe"
"== FOCUSED test_market_data ==" | Out-File $out -Encoding utf8
& $py -m unittest discover -s tests -p test_market_data.py >> $out 2>&1
if ($LASTEXITCODE -ne 0) { "1" | Out-File $exitFile -Encoding ascii; exit 1 }
"== FULL tests ==" | Out-File $out -Append -Encoding utf8
& $py -m unittest discover -s tests -p test_*.py >> $out 2>&1
if ($LASTEXITCODE -ne 0) { "2" | Out-File $exitFile -Encoding ascii; exit 2 }
"== SCAN market ==" | Out-File $out -Append -Encoding utf8
& $py scripts/scan_price_health.py >> $out 2>&1
"== DONE scan_rc=$LASTEXITCODE ==" | Out-File $out -Append -Encoding utf8
"0" | Out-File $exitFile -Encoding ascii
exit 0
