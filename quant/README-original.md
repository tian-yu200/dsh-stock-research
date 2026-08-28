# A 股与基金量化研究（AkShare + Qlib）

这是一个面向本地研究的 A 股小型量化项目，当前已打通：

完整操作说明见 [项目使用说明书](docs/项目使用说明书.md)。

```text
AkShare 多源抓取 -> 统一日线字段 -> Qlib 文件 provider
                    -> Alpha158 特征 -> LightGBM 训练与预测
                    -> TopkDropout 策略回测 -> CSV 结果
```

项目保留两套互不覆盖的 Qlib 数据：

- `qlib_data/cn_data`：Qlib 官方历史 A 股数据，用于原有沪深 300 回测。
- `qlib_data/akshare_data`：由本项目的 AkShare CSV 构建，用于近期行情端到端验证。

## 核心工作流

### 1. 行情采集

`scripts/fetch_stock_data.py` 抓取默认 5 只 A 股，数据源按东财、腾讯、新浪依次降级；`scripts/fetch_fund_data.py` 抓取 3 只 ETF 和 2 只场外基金。

股票与 ETF 日线都会经过 `scripts/market_data.py`，统一为：

```text
date, open, high, low, close, volume, amount, turnover
```

归一化过程负责列名映射、日期解析、数值转换、升序排列、重复日期去重和 OHLCV 完整性校验。场外基金净值会保存为 CSV，但不会进入当前 Alpha158 股票池。

### 2. 构建 Qlib provider

`scripts/build_qlib_data.py` 调用 `scripts/qlib_export.py`，读取：

- `data/stocks/*.csv` -> `stocks` 市场分组
- `data/funds/etf_*.csv` -> `etfs` 市场分组
- `data/benchmarks/*.csv`（可选）-> `benchmarks` 市场分组

转换器生成 Qlib 所需的交易日历、标的列表和逐字段二进制特征，并先写 staging 目录，再原子替换 `qlib_data/akshare_data`。已有 provider 不会在构建中途变成半成品，官方 `cn_data` 也不会被修改。

当前导出的字段包括 `open/high/low/close/vwap/volume/amount/turnover/change/factor`。`validate_provider()` 会检查日历、标的区间、特征文件长度和起始索引。

### 3. Alpha158 与模型训练

`scripts/run_backtest.py` 根据 YAML 初始化对应 provider，并建立：

- `Alpha158`：从 OHLCV/VWAP 生成 158 个量价特征。
- `DatasetH`：按训练、验证、测试时间段切分数据。
- `LightGBM`：在训练集拟合，在验证集早停，在测试集产生每日选股分数。

默认 AkShare 配置位于 `configs/qlib_akshare_config.yaml`，股票池使用 `stocks` 分组，即当前 5 只股票。预测写入 `results/akshare/predictions.csv`。

### 4. 策略回测

测试集预测交给 Qlib 的 `TopkDropoutStrategy`：每日持有预测分数最高的 3 只股票，并允许替换 1 只。基准为 `SH510300`，交易价格、手续费、印花税和最低费用由 YAML 的 `exchange_kwargs` 控制。

输出包括：

- `portfolio_report.csv`：每日账户、收益、换手、成本、现金和基准收益。
- `risk_analysis.csv`：扣费前后超额收益、信息比率和最大回撤。
- `run_config.yaml`：本次实际使用的配置快照。

## 运行方式

项目命令均可直接使用虚拟环境中的 Python，无需先激活环境。

完整执行采集、转换和回测：

```powershell
.\venv\Scripts\python.exe scripts\run_pipeline.py
```

使用已有 CSV 重建 provider 并回测：

```powershell
.\venv\Scripts\python.exe scripts\run_pipeline.py --skip-fetch
```

只采集并转换，不运行模型：

```powershell
.\venv\Scripts\python.exe scripts\run_pipeline.py --skip-backtest
```

也可以分步执行：

```powershell
.\venv\Scripts\python.exe scripts\fetch_stock_data.py
.\venv\Scripts\python.exe scripts\fetch_fund_data.py
.\venv\Scripts\python.exe scripts\build_qlib_data.py --replace
.\venv\Scripts\python.exe scripts\demo_akshare_qlib.py
.\venv\Scripts\python.exe scripts\run_backtest.py --config configs\qlib_akshare_config.yaml
```

原有官方数据回测入口保持不变：

```powershell
.\venv\Scripts\python.exe scripts\run_backtest.py
```

它默认读取 `configs/qlib_config.yaml` 和 `qlib_data/cn_data`，结果仍写入 `results/`。

## 验证

运行标准库测试和语法检查：

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m compileall -q scripts tests
```

测试覆盖统一字段、日期索引、缺失字段、重复日期、ETF 降级路径、证券代码映射、临时 provider 构建、二进制契约和 Qlib 原生读取。`demo_akshare_qlib.py` 是只读检查，不会重新抓取行情或覆盖正式 CSV。

## 数据约定与局限

- AkShare 股票和 ETF 输入使用前复权行情；导出时写入 `factor=1.0`。这适合当前研究原型，但不是完整的原始价格与复权因子体系。
- 各 AkShare 适配器的成交量单位不完全一致，不能稳定地用 `amount / volume` 计算统一 VWAP；当前 `vwap` 使用 `close` 作为兼容 Alpha158 的代理值。
- ETF 数据源可能不提供 `amount` 或 `turnover`，这两个字段允许为空；OHLCV 必须完整。
- 当前横截面只有 5 只股票，样本很小。回测用于证明工程链路可运行，不足以支持可靠的策略收益结论。
- Qlib 官方样例数据当前覆盖 `1999-11-10` 至 `2020-09-25`；AkShare provider 的范围由本地 CSV 决定。
- Qlib 可能提示 Gym 已停止维护，以及 CatBoost、XGBoost、PyTorch 未安装；本工作流只使用 LightGBM，这些可选组件提示不影响运行。

## 主要配置入口

- 股票池：`scripts/fetch_stock_data.py` 中的 `DEFAULT_POOL`。
- ETF 与场外基金池：`scripts/fetch_fund_data.py` 中的 `ETF_POOL`、`OTC_POOL`。
- AkShare 训练区间、模型、策略与成本：`configs/qlib_akshare_config.yaml`。
- 官方历史回测：`configs/qlib_config.yaml`。
- 新增基准 CSV：放入 `data/benchmarks/`，再重建 provider。
