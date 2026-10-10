# A 股与基金量化研究助手

面向个人本地研究的 A 股 / ETF / 场外基金量化研究系统。把行情采集、特征工程、模型训练、回测验证和组合约束串成一条可复现的流水线，并内置**发布门槛**：模型指标不达标时不输出个股买入权重，只给现金/观察结论。

> **这不是交易系统。** 不连接券商、不读取账户、不自动下单，只输出百分比目标权重和调仓比例。所有结论仅供研究参考，不构成投资建议。

## 它解决什么问题

个人做 A 股量化研究容易踩三类坑，本项目针对性地做了处理：

1. **数据不可信** —— 多数据源抓取（东财 / 腾讯 / 新浪依次降级）、统一字段、价格守卫、源级熔断、断点续抓；抓取状态持久化到 SQLite，进程重启后接着跑，不会从零重来。
2. **回测虚高** —— 信号由当日收盘价算出、系统收盘后才运行，因此股票回测从信号**次日收盘**入场（与 20 日预测标签同一区间），入场日封涨停或停牌的候选直接跳过、不补位；成本按换手 × `transaction_cost_bps`（默认 10bp，约等于双边佣金 + 卖出印花税，不含滑点）扣除。
3. **过拟合被当成有效信号** —— 用 RankIC、RankIC IR、分组超额收益、超额回撤等门槛把关，不达标就拒绝发布并保留上一份成功快照；多窗口 IC 稳定性由离线脚本 `quant/scripts/stability_check.py` 评估，结果随快照附带，不阻断发布。

## 研究流水线

```text
AkShare 多源抓取 → 统一日线字段 → Qlib 文件 provider
                 → Alpha158 特征 → LightGBM 训练与预测
                 → Top-k 重叠持有期组合回测 → 门槛判定 → 目标权重
```

三条独立路径：

- **股票**：Alpha158 + LightGBM，只有通过全部模型门槛才生成个股目标权重。
- **ETF / 场外基金**：走独立的趋势、收益、波动、回撤、基准相对表现和同类比较路径。
- **组合**：统一执行现金比例、类别敞口、单标的权重和换手限制。

## 快速开始

需要 Python 3.12。研究代码跨平台，`bin/` 与 `tasks/` 下的 PowerShell 入口依赖 Windows。

```powershell
# 1. 建立虚拟环境并安装依赖
python -m venv quant\.venv
quant\.venv\Scripts\python.exe -m pip install -r quant\requirements-base.txt

# 2. 抓取行情并构建 Qlib provider
quant\.venv\Scripts\python.exe quant\scripts\fetch_stock_data.py
quant\.venv\Scripts\python.exe quant\scripts\build_qlib_data.py --replace

# 3. 跑一次定向研究（指定股票代码）
quant\.venv\Scripts\python.exe quant\scripts\run_assistant_research.py `
    --mode on_demand --asset-scope stock --symbols 600519
```

也可以走 PowerShell 入口：

```powershell
& .\bin\run-research.ps1 -Mode on_demand -AssetScope stock -Symbols 600519 000001
```

省略 `-Symbols` 时按 `quant\data\universe.json` 扫描完整代码池。入口只接受固定模式、固定资产范围和最多 100 个六位代码，不接受输出目录、Python 路径或任意命令参数。

### 研究深度

所有入口共用三级深度：

| 深度 | 行为 |
|---|---|
| `quick` | 只读缓存 CSV 与既有 provider/模型缓存，不刷新行情；秒级返回，数据可能陈旧 |
| `standard` | 缺省值：定向请求刷新指定代码，全市场请求按代码池推进 |
| `full` | 忽略定向代码，刷新整个资产代码池并重建 provider |

注意冷启动成本：provider 重建会使模型缓存失效，重建后第一次股票分析需要完整训练一次模型（实测 8 线程约 26 分钟）。完成后同周期内后续请求为秒级。

### 自动任务

注册每周一至周五 18:30（北京时间）的 Windows 计划任务：

```powershell
& .\tasks\register-task.ps1    # 注册
& .\tasks\unregister-task.ps1  # 注销
```

任务先通过交易日历判断是否为交易日，非交易日跳过；当月最后一个有效交易日运行 `monthly`，其他交易日运行 `daily`。

## 目录结构

```text
.
├── quant/
│   ├── assistant/     研究服务核心（研究管线、模型、风险、组合、路径守卫）
│   ├── scripts/       抓取、provider 构建、回测、任务入口
│   ├── configs/       YAML 配置（抓取、门槛、模型、数据集）
│   ├── mcp/           MCP server
│   ├── tests/         单元测试与回归测试
│   └── docs/          研究管线不变量
├── bin/               PowerShell 启动与手动研究入口
├── tasks/             Windows 任务计划程序脚本
└── docs/              运行说明、安全边界、验收说明、重构计划
```

运行时数据（`quant/data/`、`quant/results/`、`quant/logs/`、`quant/qlib_data/`）和虚拟环境不入库。

## MCP 接口

研究能力以 MCP 工具形式暴露，可接入支持 MCP 的 agent 客户端：

```text
run_research           提交研究请求（模式 / 资产范围 / 深度 / 代码）
get_current_snapshot   读取当前已发布快照
get_rolling_scan       读取滚动初选池
get_run_status         查询运行状态与阶段耗时
list_notifications     列出通知
ack_notification       确认通知
```

工具只接受固定模式、固定资产范围和六位代码，**没有路径、命令或输出目录参数**——不会执行任意 PowerShell 或任意 Python 文件。MCP 使用独立虚拟环境，协议 JSON 写 stdout，诊断日志写 stderr。

## 门槛与安全边界

发布判据位于 `quant/configs/assistant.yaml` 的 `gate` 段：

| 门槛 | 缺省值 | 含义 |
|---|---|---|
| `min_mean_rank_ic` | 0.02 | 平均 RankIC 下限 |
| `min_rank_ic_ir` | 0.10 | RankIC 信息比下限 |
| `min_top_bottom_excess_return` | 0.00 | 分组多空超额收益下限 |
| `max_excess_test_drawdown` | 0.15 | 策略回撤超出基准的容忍度 |
| `max_ic_std` | 0.02 | 多窗口 IC 标准差上限（稳定性检查启用时生效） |

设计上的几个硬约束：

- **宁缺毋滥**：剔除 ST/*ST/退市整理、上市不满约 60 个交易日的次新、20 日均成交额低于 1000 万元的低流动性标的。
- **海选原则**：个股目标权重只能由全市场筛选的排名给出；定向单票研究只输出信号与门槛，不产生买入权重。
- **失败即保留**：失败、超时或校验失败时保留上一份成功快照并写入通知，不会用坏结果覆盖好结果。
- **门槛只收紧不放宽**：配置内保留了每次门槛调整的理由与实证依据；调整门槛必须同步修改测试。

## 测试

```powershell
quant\.venv\Scripts\python.exe -m pytest quant\tests -q
```

测试覆盖数据源降级与熔断、统一字段与复权、provider 构建与二进制契约、模型门槛、标签泄漏检查、单位守卫、组合约束、结果原子发布、通知、MCP 参数校验和路径越界拒绝。

## 已知局限

- AkShare 接口稳定性受上游影响；抓取侧有降级、熔断和续跑保护，但不保证数据源永远可用。
- 横截面样本规模决定结论强度。小样本回测只能证明工程链路可通，不足以支撑策略收益结论。
- 门槛参数基于特定市场阶段调参，换市场或换周期需要重新验证。
- 场外基金只使用净值研究，不输出盘中成交价和精确股数。
- 股票回测未建模跌停卖不出与 ST 股 ±5% 涨跌幅；成本不含滑点与冲击，对低价股偏乐观。

## 文档

- [`docs/运行说明.md`](docs/运行说明.md) —— 部署与运行细节
- [`docs/安全边界.md`](docs/安全边界.md) —— 工具面与写入边界
- [`docs/数据守卫说明.md`](docs/数据守卫说明.md) —— 数据质量与价格守卫
- [`quant/docs/研究管线不变量.md`](quant/docs/研究管线不变量.md) —— 修改研究代码前必读的实证不变量与禁忌
- [`docs/重构计划.md`](docs/重构计划.md) —— 架构演进计划

## License

[MIT](LICENSE)
