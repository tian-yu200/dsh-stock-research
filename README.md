# DSH 股票基金私人研究助手

本目录把已安装的 DSH 运行时与量化研究项目隔离开，所有项目配置、结果、日志、数据和 Skill 都位于 `D:\dsh-股票`。

## 目录

```text
D:\dsh-股票\
  .dsh\                         DSH_HOME、profile patch、preset 和 Skill
  quant\                        独立 Python 环境、数据、Qlib provider、研究代码
  bin\                          启动 DSH 和手动研究入口
  tasks\                        Windows 任务计划程序脚本
  docs\                         运行、验收和安全边界说明
```

## 启动

在 PowerShell 中执行：

```powershell
& 'D:\dsh-股票\bin\start-stock-assistant.ps1'
```

启动脚本设置：

```text
DSH_HOME=D:\dsh-股票\.dsh
DSH_CWD=D:\dsh-股票
```

它调用已安装的 `D:\deepseek-harness\node_modules\.bin\dsh.cmd`，不会修改 DSH 安装目录。

## 手动研究

```powershell
& 'D:\dsh-股票\bin\run-research.ps1' -Mode on_demand -AssetScope stock -Symbols 600519 000001
& 'D:\dsh-股票\bin\run-research.ps1' -Mode on_demand -AssetScope etf -Symbols 510300
& 'D:\dsh-股票\bin\run-research.ps1' -Mode on_demand -AssetScope otc_fund -Symbols 110020
```

省略 `-Symbols` 会按 `quant\data\universe.json` 扫描完整股票、ETF 或场外基金代码池。入口只接受固定模式、固定资产范围和最多 100 个六位代码，不接受输出目录、Python 路径或命令参数。

## 自动任务

以当前 Windows 用户注册每周一至周五 18:30 的任务：

```powershell
& 'D:\dsh-股票\tasks\register-task.ps1'
```

任务先通过 AkShare 交易日历接口判断是否为交易日，并将成功日历原子缓存到 `quant\data\benchmarks\trade_calendar.csv`；在线接口失败时只使用该本地缓存。非交易日跳过；当月最后一个有效交易日运行 `monthly`，其他交易日运行 `daily`。注销：

```powershell
& 'D:\dsh-股票\tasks\unregister-task.ps1'
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

MCP server 通过 `D:\dsh-股票\quant\.venv` 的 Python 启动，stdout 只用于 MCP 协议，日志写 stderr。

## 结果

每次运行写入：

```text
D:\dsh-股票\quant\results\assistant\runs\YYYY\MM\DD\<run_id>\
```

目录包含 manifest、数据事实、模型信号、规则门槛、决策支持、配置快照、摘要和日志。只有成功且校验通过的运行才原子更新：

```text
D:\dsh-股票\quant\results\assistant\current.json
```

失败、超时和校验失败保留上一份成功快照，并写入 DSH 收件箱。

## 研究边界

股票使用 Alpha158 + LightGBM，只有通过 RankIC、RankIC IR、分组超额收益和回撤门槛才会生成个股目标权重。ETF 与场外基金走独立的趋势、收益、波动、回撤、基准相对表现和同类比较路径。组合结果统一执行现金比例、类别敞口、单标的权重和换手限制。

首期只生成百分比目标权重和调仓比例，不连接真实券商、不自动下单。场外基金只使用净值研究，不输出盘中成交价和精确股数。详细说明见 `docs\安全边界.md`。
