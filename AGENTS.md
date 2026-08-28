# Workspace Stability Rule

When working in this repository, treat any command that may run longer than 30 seconds, scan the full market, recurse through large data/log directories, run a full test suite, or emit more than 200 lines / 20 KB as a heavy operation. Apply this procedure as one hard rule:

1. Keep foreground work bounded. Use focused file reads and queries (`rg`, `Select-Object -First`, `Get-Content -Tail`) and cap returned output at 200 lines or 20 KB. Summarize counts and samples instead of returning complete JSONL sessions, logs, recursive listings, SQLite rows, or full source files.
2. Before a heavy operation, inspect existing `python`, `powershell`, and project worker processes. Do not overlap a full-market fetch, a full test suite, or another heavy project command.
3. Run every heavy operation outside the interactive Codex execution stream with `Start-Process -WindowStyle Hidden`, redirect stdout and stderr to timestamped files under `quant/logs/codex-runs/`, and record its PID. The operation must use the application's existing checkpoint/state database; never restart a full-market job from zero when a resumable job exists.
4. Observe heavy work only through bounded status checks: process state, status JSON/SQLite aggregates, and at most the last 80 log lines. Poll no more often than every 15 seconds and never reread a growing log in full.
5. Run focused tests first. Start a full suite only after focused tests pass, and run that full suite as a logged heavy operation.
6. On `aborted`, response-stream retries, a user interruption, or an unresponsive tool call, stop launching commands. First determine whether the recorded process is still alive. If alive, leave it running and resume bounded polling; if dead, preserve logs and checkpoints, diagnose the last bounded log tail, and resume only the smallest failed step.

This rule exists because prior sessions repeatedly combined 47-60 KB tool results, recursive/full-file reads, foreground long-running tests or market fetches, and response-stream interruption. Codex logs recorded repeated stream retries and an interrupted task that did not exit gracefully; Windows event logs showed no Codex crash and there was no evidence of OOM, so avoid asserting memory exhaustion without new evidence.

## Interactive Boundary Rule

The interactive Codex execution stream is a control plane only. Keep it short and detachable:

- Treat any test suite, full-market operation, command expected to exceed 10 seconds, command that starts child processes or threads, and command that may emit more than 100 lines or 10 KB as detached heavy work. Launch it with `Start-Process -WindowStyle Hidden`, redirect stdout and stderr to timestamped files under `quant/logs/codex-runs/`, record the PID and an exit-code file, then return immediately.
- Never run heavy work in the foreground and never wait for it from the interactive stream: do not use `Wait-Process`, a blocking `Start-Process` wrapper, `Start-Sleep` in the same launch command, `yield_time_ms` above 10000, or a command that keeps the tool call attached until the child exits.
- Poll a recorded PID only in a separate bounded command no more often than every 15 seconds. Read process state plus at most the last 80 lines of each log; do not stream, reread, or parse a complete session, JSONL file, SQLite table, or growing log.
- After `aborted`, `KeyboardInterrupt`, `0xC000013A`, a tool timeout, a response-stream retry, or a blank/unresponsive window, stop launching commands. Check the recorded PID once. If it is alive, leave it running and only perform bounded polling; if it is dead, preserve its logs, inspect the bounded tail, and isolate the smallest failed step in a fresh detached process before any rerun.
- Do not rerun the same full suite or market job after an interruption until the previous PID, exit code, and bounded log tail have been recorded. Reuse checkpoints and existing job IDs; never start a second worker wave to compensate for an interrupted UI session.
- Make detached Python output encoding-safe before launch: set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`. Console/status JSON written to redirected logs must use ASCII-safe serialization such as `ensure_ascii=True` and must contain bounded aggregates rather than full symbol arrays or complete job metadata.
- Evidence for this rule: three complete-suite attempts stopped at `test_fetch_fund_worker.FundWorkerTests.test_main_claims_only_one_worker_wave_at_a_time` with `KeyboardInterrupt` and exit code `0xC000013A`, while that test passed alone in 0.004 seconds; no Codex Crashpad report or Windows Codex/Electron crash event was present. Treat this pattern as execution-stream interruption or process-tree teardown, not as proof of an application OOM.
