# Workspace Stability Rule

## 研究管线不变量（修改代码前必读）

修改 `quant/scripts/`（data_source、market_data、price_guards、fetch_*、continuous_research）或 `quant/assistant/`（research、data_layer、qlib_model）之前，**必须先阅读 `quant/docs/研究管线不变量.md`**。该文件记录 2026-09-05 大修中被实证的不变量与禁忌：市场前缀映射（920=北交所→bj）、源级熔断与确定性异常重试语义、非正价格行丢弃、价格守卫的 A/B 黄金基线、worker 预算 7200s 的依据、16GB 内存约束下的训练窗口、海选原则门槛语义（定向研究 failed 是设计行为）、以及 job 台账 stderr 误报的判读方法。违反这些规则的修改曾导致 900+ 标的卡死一天、扫描 9 小时跑不完、模型 OOM。

When working in this repository, treat any command that may run longer than 30 seconds, scan the full market, recurse through large data/log directories, run a full test suite, or emit more than 200 lines / 20 KB as a heavy operation. Apply this procedure as one hard rule:

1. Keep foreground work bounded. Use focused file reads and queries (`rg`, `Select-Object -First`, `Get-Content -Tail`) and cap returned output at 200 lines or 20 KB. Summarize counts and samples instead of returning complete JSONL sessions, logs, recursive listings, SQLite rows, or full source files.
2. Before a heavy operation, inspect existing `python`, `powershell`, and project worker processes. Do not overlap a full-market fetch, a full test suite, or another heavy project command.
3. Run every heavy operation outside the interactive Codex execution stream with `Start-Process -WindowStyle Hidden`, redirect stdout and stderr to timestamped files under `quant/logs/codex-runs/`, and record its PID. The operation must use the application's existing checkpoint/state database; never restart a full-market job from zero when a resumable job exists.
4. Observe heavy work only through bounded status checks: process state, status JSON/SQLite aggregates, and at most the last 80 log lines. Poll no more often than every 15 seconds and never reread a growing log in full.
5. Run focused tests first. Start a full suite only after focused tests pass, and run that full suite as a logged heavy operation.
6. On `aborted`, response-stream retries, a user interruption, or an unresponsive tool call, stop launching commands. First determine whether the recorded process is still alive. If alive, leave it running and resume bounded polling; if dead, preserve logs and checkpoints, diagnose the last bounded log tail, and resume only the smallest failed step.

This rule exists because prior sessions repeatedly combined 47-60 KB tool results, recursive/full-file reads, foreground long-running tests or market fetches, and response-stream interruption. Codex logs recorded repeated stream retries and an interrupted task that did not exit gracefully; Windows event logs showed no Codex crash and there was no evidence of OOM, so avoid asserting memory exhaustion without new evidence.

## Sandbox Limits (read before running anything)

Codex runs in a restricted Windows sandbox (account `codexsandboxoffline`, not
elevated) and cannot use the Windows Store Python that this project's venv is
built on. Concretely:

- `quant\.venv\pyvenv.cfg` has `home = C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0`
- That path is a Windows Store **App Execution Alias** — a reparse point
  (`ReparsePoint` attribute, `LinkType`/`Target` empty), not a classic symlink.
  Its real target is the versioned package dir
  `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe`.
- What the sandbox blocks is the whole Store install/alias area, not a single
  file: listing either `C:\Program Files\WindowsApps` or
  `C:\Users\<user>\AppData\Local\Microsoft\WindowsApps` returns `Access is denied`,
  and `Test-Path` on the alias `home` above also returns `Access is denied`.
  As a result `quant\.venv\Scripts\python.exe` fails with `No Python at '...'`.

**The venv is not broken.** The launcher file exists
(`Test-Path quant\.venv\Scripts\python.exe` = True); the same interpreter runs
fine from a normal shell, from `tasks\*.ps1`, and from opencode (full suite:
325 passed). Do not propose reinstalling or rebuilding the venv — that wastes a
cycle and changes nothing. (Note: "not broken" is reported, not verifiable from
inside the Codex sandbox — verifying it is exactly what the sandbox prevents.)

**What to do instead:**

- **Do not attempt to run `quant\.venv\Scripts\python.exe`.** This is not a
  launcher problem: `py.exe` itself runs, and `py.exe --list` enumerates the
  installed interpreters (Store 3.12, Store 3.11, Astral CPython 3.11.16). But
  launching *this project's venv* through any launcher still fails, because the
  venv's `base`/`home` points into the blocked Store area.
- **Do not run `pytest`** or any command that needs the project's Python.
- **Report it and hand off.** State plainly that the task needs Python and is
  therefore opencode's job. That is a normal, expected outcome — not a failure.
- Tasks that stay with Codex: reading and writing files, `rg`/`Select-String`
  searches, PowerShell that does not invoke the project's Python, and bounded
  status checks of JSON/SQLite aggregates.

One more Codex-specific limit: a turn that chains many tool calls in a row can
trip `HTTP 400 reasoning_content_missing` upstream and kill the turn. Keep
turns short — one or two steps — and expect multi-step exploration to be handed
to opencode.

## Interactive Boundary Rule

The interactive Codex execution stream is a control plane only. Keep it short and detachable:

- Treat any test suite, full-market operation, command expected to exceed 10 seconds, command that starts child processes or threads, and command that may emit more than 100 lines or 10 KB as detached heavy work. Launch it with `Start-Process -WindowStyle Hidden`, redirect stdout and stderr to timestamped files under `quant/logs/codex-runs/`, record the PID and an exit-code file, then return immediately.
- Never run heavy work in the foreground and never wait for it from the interactive stream: do not use `Wait-Process`, a blocking `Start-Process` wrapper, `Start-Sleep` in the same launch command, `yield_time_ms` above 10000, or a command that keeps the tool call attached until the child exits.
- Poll a recorded PID only in a separate bounded command no more often than every 15 seconds. Read process state plus at most the last 80 lines of each log; do not stream, reread, or parse a complete session, JSONL file, SQLite table, or growing log.
- After `aborted`, `KeyboardInterrupt`, `0xC000013A`, a tool timeout, a response-stream retry, or a blank/unresponsive window, stop launching commands. Check the recorded PID once. If it is alive, leave it running and only perform bounded polling; if it is dead, preserve its logs, inspect the bounded tail, and isolate the smallest failed step in a fresh detached process before any rerun.
- Do not rerun the same full suite or market job after an interruption until the previous PID, exit code, and bounded log tail have been recorded. Reuse checkpoints and existing job IDs; never start a second worker wave to compensate for an interrupted UI session.
- Make detached Python output encoding-safe before launch: set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`. Console/status JSON written to redirected logs must use ASCII-safe serialization such as `ensure_ascii=True` and must contain bounded aggregates rather than full symbol arrays or complete job metadata.
- Evidence for this rule: three complete-suite attempts stopped at `test_fetch_fund_worker.FundWorkerTests.test_main_claims_only_one_worker_wave_at_a_time` with `KeyboardInterrupt` and exit code `0xC000013A`, while that test passed alone in 0.004 seconds; no Codex Crashpad report or Windows Codex/Electron crash event was present. Treat this pattern as execution-stream interruption or process-tree teardown, not as proof of an application OOM.
