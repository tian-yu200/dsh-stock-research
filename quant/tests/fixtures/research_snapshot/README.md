# 研究引擎快照测试固定数据集（阶段 0 安全网）

供 `tests/test_research_snapshot.py` 使用。数据集由脚本一次性生成，之后作为
静态文件提交，测试只复制、不重新生成，保证跨机器逐字节一致。

## 组成（7 个 CSV，各 300 个交易日，截至 2026-08-25）

| 文件 | 资产类型 | 走势 |
|---|---|---|
| `stocks/600519.csv` | 股票 | 上升（10 → 20 线性） |
| `stocks/000001.csv` | 股票 | 上升（与 600519 完全相同序列） |
| `stocks/300750.csv` | 股票 | 下降（20 → 10 线性） |
| `funds/etf_510300.csv` | ETF | 上升；**同时充当 510300 基准**（编排层用 `_load_asset("etf", "510300")` 读基准） |
| `funds/etf_159915.csv` | ETF | 下降 |
| `funds/fund_110020.csv` | 场外基金 | 上升（仅 date+close 净值序列） |
| `funds/fund_161725.csv` | 场外基金 | 下降（仅 date+close 净值序列） |

## 生成规则（手工可核验）

- 日期：`pd.bdate_range(end="2026-08-25", periods=300)`（纯工作日）。
- 上升序列：`np.linspace(10.0, 20.0, 300)`；下降序列：其倒序。
- K 线列：`open=close*0.99, high=close*1.01, low=close*0.98,
  volume=1000..1299, amount=close*volume, turnover=1.0`。
- 由此可手算验证：上升资产全序列最大回撤 = 0、当前回撤 = 0；
  下降资产最大回撤 = 当前回撤 = 10/20 − 1 = −0.5。

## 再生方式（仅在需要扩展资产时使用，勿覆盖现有文件）

```python
import numpy as np, pandas as pd
dates = pd.bdate_range(end="2026-08-25", periods=300)
close = np.linspace(10.0, 20.0, len(dates))  # 下降取 [::-1]
```
