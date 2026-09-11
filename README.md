# DSH 股票基金私人研究助手

本目录把已安装的 DSH 运行时与量化研究项目隔离开，所有项目配置、结果、日志、数据和 Skill 都位于 `D:\dsh-gupiao`。

## 目录

```text
D:\dsh-gupiao\
  .dsh\                         DSH_HOME、profile patch、preset 和 Skill
  quant\                        独立 Python 环境、数据、Qlib provider、研究代码
  bin\                          启动 DSH 和手动研究入口
  tasks\                        Windows 任务计划程序脚本
  docs\                         运行、验收和安全边界说明
```

## 启动

在 PowerShell 中执行：

```powershell
& 'D:\dsh-gupiao\bin\start-stock-assistant.ps1'
```

启动脚本设置：

```text
DSH_HOME=D:\dsh-gupiao\.dsh
DSH_CWD=D:\dsh-gupiao
```

它调用已安装的 `D:\deepseek-harness\node_modules\.bin\dsh.cmd`，不会修改 DSH 安装目录。

## 手动研究

```powershell
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope stock -Symbols 600519 000001
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope etf -Symbols 510300
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope otc_fund -Symbols 110020
```

省略 `-Symbols` 会按 `quant\data\universe.json` 扫描完整股票、ETF 或场外基金代码池。入口只接受固定模式、固定资产范围和最多 100 个六位代码，不接受输出目录、Python 路径或命令参数。

### 研究深度 depth

所有入口（MCP `run_research`、CLI `--depth`、`run-research.ps1 -Depth`）共用三级深度：

```text
quick      只读缓存 CSV 与既有 Qlib provider/模型缓存，不刷新任何行情；
           秒级返回，数据可能陈旧（结果中带 data_end_date 与 stale_allowed_by_depth 标记）。
standard   缺省值：daily/monthly 定时任务与带 -Symbols 的定向请求刷新指定代码，
           全市场请求按代码池推进。
full       忽略定向代码，刷新整个所选资产代码池并重建 provider；不能与 symbols 组合。
```

`quant\configs\assistant.yaml` 的 `research.default_depth` 只影响 on_demand 请求的缺省深度；daily/monthly 定时任务始终按 standard（全量维护）执行。

注意冷启动：provider 每次重建（全市场维护后）都会使模型缓存失效，**重建后的第一笔股票分析**（quick 或 standard）需要完整训练一次模型——实测 8 线程约 26 分钟（训练无检查点，属每周期一次性成本），完成后缓存随即可用，同周期后续请求为秒级（实测缓存命中约 8 秒）。定向任务一次 worker 预算由 `fetch.targeted_time_budget_seconds`（缺省 3600 秒）保证，研究阶段也不会被全市场波次预算（`fetch.batch_time_budget_seconds`，600 秒）截断。另外研究阶段有全局锁：上一笔研究未结束时提交的新请求会失败，请等完成通知后再发下一笔。

`get_current_snapshot` 返回只读 `freshness` 块：按交易日目标计算所选切片的 completed/pending/stale/coverage，与抓取队列判定一致。数据够新时先答 `quick`，`pending>0` 时再用 `standard` 刷新指定代码，即可把"询问到出结果"压到最短。

抓取波次支持并行通道（`fetch.parallel_asset_fetch`，缺省开启）：股票脚本与基金脚本访问不同数据源，两通道并行不提高同一端点请求速率；etf 与 otc_fund 共用基金通道内部串行。全市场任务的抓取墙钟时间约等于两条通道中较慢者，而非两者之和。

每次运行的 `status.json` 记录 `depth`、排队等待 `queue_wait_seconds` 与各阶段耗时 `stage_timings`（cache_check/fetch/qlib_build/research 等）；研究 `manifest.json` 也带 `depth` 与 `stage_timings`，用于后续定位耗时。

```powershell
# 秒级看某只股票的既有结论
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope stock -Depth quick -Symbols 600519
# 刷新该股票后出结论（分钟级）
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope stock -Depth standard -Symbols 600519
# 全代码池维护（后台波次推进，最慢）
& 'D:\dsh-gupiao\bin\run-research.ps1' -Mode on_demand -AssetScope stock -Depth full
```

## 自动任务

以当前 Windows 用户注册每周一至周五 18:30 的任务：

```powershell
& 'D:\dsh-gupiao\tasks\register-task.ps1'
```

任务先通过 AkShare 交易日历接口判断是否为交易日，并将成功日历原子缓存到 `quant\data\benchmarks\trade_calendar.csv`；在线接口失败时只使用该本地缓存。非交易日跳过；当月最后一个有效交易日运行 `monthly`，其他交易日运行 `daily`。注销：

```powershell
& 'D:\dsh-gupiao\tasks\unregister-task.ps1'
```

## MCP 工具

股票 preset 只挂载以下固定工具：

```text
mcp__stock_research__run_research
mcp__stock_research__get_current_snapshot
mcp__stock_research__get_run_status
mcp__stock_research__list_notifications
mcp__stock_research__ack_notification
```

MCP server 通过 `D:\dsh-gupiao\quant\.venv` 的 Python 启动，stdout 只用于 MCP 协议，日志写 stderr。

## 结果

每次运行写入：

```text
D:\dsh-gupiao\quant\results\assistant\runs\YYYY\MM\DD\<run_id>\
```

目录包含 manifest、数据事实、模型信号、规则门槛、决策支持、配置快照、摘要和日志。只有成功且校验通过的运行才原子更新：

```text
D:\dsh-gupiao\quant\results\assistant\current.json
```

失败、超时和校验失败保留上一份成功快照，并写入 DSH 收件箱。

## 研究边界

股票使用 Alpha158 + LightGBM，只有通过 RankIC、RankIC IR、分组超额收益和回撤门槛才会生成个股目标权重。ETF 与场外基金走独立的趋势、收益、波动、回撤、基准相对表现和同类比较路径。组合结果统一执行现金比例、类别敞口、单标的权重和换手限制。

首期只生成百分比目标权重和调仓比例，不连接真实券商、不自动下单。场外基金只使用净值研究，不输出盘中成交价和精确股数。详细说明见 `docs\安全边界.md`。
