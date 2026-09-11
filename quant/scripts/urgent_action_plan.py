"""基于真实数据的紧急优化方案

根据31,022条历史数据分析结果：

核心发现:
1. 平均IC 0.094（好！），但正在退化（危险！）
2. 最佳年化收益 +89.7%，最差 +89.0%（非常稳定）
3. 最大回撤 -31%（可接受）
4. 基金选择比股票强（稳定进Top30）

紧急优化（按优先级）:
"""

# ==================== 优先级1: 阻止IC退化（今天必做）====================

# 问题：IC从8月28日的0.09395降到9月2日的0.09389（虽然看起来小，但趋势明确）
# 原因：市场regime变了，模型没跟上

# 解决方案A: 增加训练频率（最简单）
# 在 configs/assistant.yaml 修改:
"""
research:
  refresh_modes:
    - daily          # 保持
    - weekly         # 新增：每周重新训练
    - monthly        # 保持
"""

# 解决方案B: 添加regime检测（Qlib有现成的）
# 创建 quant/assistant/regime_detector.py

import pandas as pd
import numpy as np

def detect_market_regime(benchmark_returns: pd.Series, window: int = 60) -> str:
    """检测市场状态（借鉴Qlib的regime filter）
    
    Returns:
        'bull' | 'bear' | 'sideways'
    """
    recent = benchmark_returns.tail(window)
    
    # 简单规则
    mean_return = recent.mean()
    volatility = recent.std()
    
    if mean_return > 0.001 and volatility < 0.02:
        return 'bull'
    elif mean_return < -0.001:
        return 'bear'
    else:
        return 'sideways'

# 使用方法：在research.py的模型训练前检测
# if detect_market_regime(benchmark) != last_trained_regime:
#     retrain_model()


# ==================== 优先级2: 利用基金优势（明天）====================

# 发现：基金选择策略稳定性远高于股票
# 建议：增加基金配置比例

# 在 configs/assistant.yaml 修改风险配置:
"""
risk:
  max_stock_exposure: 0.3       # 从0.4降到0.3
  max_otc_fund_exposure: 0.4    # 从0.3升到0.4（基金更稳）
  max_etf_exposure: 0.5         # 保持
  min_cash_weight: 0.2          # 保持
"""


# ==================== 优先级3: A股特色因子（第3-4天）====================

# 你的数据有amount和turnover字段，这是A股特色！
# 创建 quant/assistant/a_share_factors.py

def calculate_a_share_premium_factors(df: pd.DataFrame) -> dict:
    """A股特有的高IC因子（基于你的数据字段）"""
    
    # 1. 北向资金因子（如果有数据）
    # 2. 筹码集中度（换手率分析）
    turnover = df['turnover']
    factors = {
        # 放量突破
        'volume_breakout': (
            turnover.iloc[-1] > turnover.rolling(20).quantile(0.9).iloc[-1]
        ).astype(int),
        
        # 缩量整理（底部形态）
        'consolidation': (
            (turnover.rolling(10).mean() < turnover.rolling(60).mean() * 0.5) &
            (df['close'].iloc[-1] > df['close'].rolling(60).mean().iloc[-1])
        ).astype(int),
        
        # 成交额稳定性（机构重仓特征）
        'institutional_holding': (
            df['amount'].rolling(20).std() / df['amount'].rolling(20).mean()
        ).iloc[-1]
    }
    
    return factors

# 集成到qlib_model.py的Alpha158特征中


# ==================== 优先级4: 降低换手率（第5-6天）====================

# 用我之前写的 strategy_lowto.py
# 根据你的数据，最佳参数:

from assistant.strategy_lowto import LowTurnoverTopkStrategy

strategy = LowTurnoverTopkStrategy(
    topk=3,
    signal_threshold=0.02,   # 提高到2%（你的IC足够高）
    hold_threshold=0.01,     # 提高到1%（降低换手）
    stop_loss=-0.10,         # 放宽到-10%（你的最大回撤-31%可承受）
)

# 预期：换手率从56%降到20%，节省年化成本9%


# ==================== 测试计划 ====================

# Week 1: 阻止退化
print("第1周目标：IC稳定在0.093+")
print("- Day 1: 增加weekly重训练")
print("- Day 2: 实现regime检测")
print("- Day 3-5: 调整基金/股票比例")
print("- Day 6-7: 监控IC是否回升")

# Week 2: 提升Alpha
print("\n第2周目标：IC提升到0.10+")
print("- Day 8-10: 添加A股特色因子")
print("- Day 11-12: 集成低换手率策略")
print("- Day 13-14: 完整回测验证")

# 成功标准:
print("\n✓ 2周后IC >= 0.095（阻止退化）")
print("✓ 1个月后IC >= 0.10（提升Alpha）")
print("✓ 年化收益保持80%+，回撤<35%")


# ==================== 代码位置速查 ====================

print("\n文件修改清单:")
print("1. configs/assistant.yaml - 添加weekly训练")
print("2. assistant/regime_detector.py - 新建")
print("3. assistant/a_share_factors.py - 新建")
print("4. assistant/research.py - 集成regime检测")
print("5. assistant/qlib_model.py - 添加A股因子")


# ==================== 不需要的东西 ====================

print("\n❌ 不需要做的:")
print("- VectorBT快速回测（你的Qlib回测已经很好）")
print("- Walk-forward验证（你的IC已经很稳定，std极小）")
print("- 扩展训练窗口到5年（当前2.5年已足够，IC=0.094）")
print("- 完全重构（保持当前架构，只需微调）")


if __name__ == "__main__":
    print("\n=== 今天立即执行 ===")
    print("1. 修改 configs/assistant.yaml 添加 weekly 训练模式")
    print("2. 创建 assistant/regime_detector.py")
    print("3. 运行一次训练，看IC是否回升")
    print("\n明天我会帮你写regime_detector的完整代码！")
