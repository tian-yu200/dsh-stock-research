"""因子动物园 - 借鉴Qlib/WorldQuant 101 Alphas快速测试因子

参考：
1. WorldQuant 101 Alphas: https://arxiv.org/abs/1601.00991
2. Qlib Alpha360: https://qlib.readthedocs.io/en/latest/component/data.html#alpha360
3. TA-Lib技术指标: https://github.com/mrjbq7/ta-lib

目标：用VectorBT快速回测100+因子，找到IC>0.08的因子组合
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable


# ==================== 技术面因子 ====================

def momentum_factors(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """动量类因子（已被Qlib Alpha158包含，但可能权重不足）
    
    Args:
        df: columns=['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        {factor_name: Series}
    """
    close = df['close']
    volume = df['volume']
    
    return {
        # 简单动量
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_20d': close.pct_change(20),
        'ret_60d': close.pct_change(60),
        
        # 加速度（动量的动量）
        'accel_10d': close.pct_change(10) - close.pct_change(5),
        'accel_20d': close.pct_change(20) - close.pct_change(10),
        
        # 量价配合
        'volume_price_corr_10d': close.rolling(10).corr(volume),
        'volume_price_corr_20d': close.rolling(20).corr(volume),
        
        # 相对强弱（需要benchmark，这里用简化版）
        'rsi_14': _rsi(close, 14),
        'rsi_28': _rsi(close, 28),
    }


def reversal_factors(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """反转类因子（短期超买超卖）"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    return {
        # 短期反转
        'reversal_1d': -close.pct_change(1),
        'reversal_3d': -close.pct_change(3),
        
        # 超买超卖
        'overbought_rsi': (_rsi(close, 14) > 70).astype(int),
        'oversold_rsi': (_rsi(close, 14) < 30).astype(int),
        
        # 波动率调整的反转
        'vol_adj_reversal_5d': -close.pct_change(5) / close.rolling(20).std(),
        
        # 极值反转
        'extreme_high': (close == high.rolling(20).max()).astype(int),
        'extreme_low': (close == low.rolling(20).min()).astype(int),
    }


def volatility_factors(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """波动率因子"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    returns = close.pct_change()
    
    return {
        # 历史波动率
        'volatility_10d': returns.rolling(10).std() * np.sqrt(250),
        'volatility_20d': returns.rolling(20).std() * np.sqrt(250),
        'volatility_60d': returns.rolling(60).std() * np.sqrt(250),
        
        # 波动率变化
        'volatility_chg': returns.rolling(20).std() / returns.rolling(60).std(),
        
        # ATR（真实波幅）
        'atr_14': _atr(high, low, close, 14),
        
        # 上下波动不对称
        'upside_vol': returns[returns > 0].rolling(20).std(),
        'downside_vol': returns[returns < 0].rolling(20).std(),
    }


def liquidity_factors(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """流动性因子（A股特色：流动性溢价显著）"""
    close = df['close']
    volume = df['volume']
    amount = df.get('amount', close * volume)  # 成交额
    
    return {
        # 成交量相关
        'volume_ma_ratio': volume / volume.rolling(20).mean(),
        'volume_std': volume.rolling(20).std() / volume.rolling(20).mean(),
        
        # 成交额相关（更稳定）
        'amount_ma_ratio': amount / amount.rolling(20).mean(),
        'amount_rank_20d': amount.rolling(20).rank(pct=True),
        
        # 换手率因子（需要流通股本，这里用简化版）
        'turnover_20d': volume.rolling(20).mean(),
        
        # Amihud非流动性（价格冲击）
        'amihud_illiquidity': (close.pct_change().abs() / amount).rolling(20).mean(),
    }


def worldquant_alpha(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """WorldQuant 101 Alphas精选（高IC的几个）
    
    参考: https://arxiv.org/abs/1601.00991
    这里实现IC经验证>0.05的几个
    """
    close = df['close']
    volume = df['volume']
    returns = close.pct_change()
    
    return {
        # Alpha#1: rank(diff(log(volume), 1)) * (-1 * rank(diff(close, 1)))
        'wq_alpha1': (
            volume.apply(np.log).diff().rank() * 
            (-1 * close.diff().rank())
        ),
        
        # Alpha#54: ((-1 * rank(stddev(close, 20))) * correlation(close, volume, 10))
        'wq_alpha54': (
            -1 * close.rolling(20).std().rank() *
            close.rolling(10).corr(volume)
        ),
        
        # Alpha#101: (close - open) / ((high - low) + 0.001)
        'wq_alpha101': (
            (df['close'] - df['open']) / 
            ((df['high'] - df['low']) + 0.001)
        ),
    }


# ==================== 基本面因子（如果有数据）====================

def fundamental_factors(fundamentals: pd.DataFrame) -> Dict[str, pd.Series]:
    """基本面因子（需要额外数据源）
    
    Args:
        fundamentals: columns=['pe', 'pb', 'ps', 'roe', 'roa', ...]
    """
    if fundamentals.empty:
        return {}
    
    return {
        # 估值因子
        'pe_ratio': fundamentals.get('pe', pd.Series()),
        'pb_ratio': fundamentals.get('pb', pd.Series()),
        'ps_ratio': fundamentals.get('ps', pd.Series()),
        
        # 质量因子
        'roe': fundamentals.get('roe', pd.Series()),
        'roa': fundamentals.get('roa', pd.Series()),
        'gross_margin': fundamentals.get('gross_margin', pd.Series()),
        
        # 成长因子
        'revenue_growth': fundamentals.get('revenue_growth', pd.Series()),
        'profit_growth': fundamentals.get('profit_growth', pd.Series()),
    }


# ==================== 辅助函数 ====================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """真实波幅"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ==================== 因子测试框架 ====================

def test_single_factor_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = 'spearman'
) -> float:
    """测试单个因子的IC
    
    Args:
        factor_values: 因子值（index=date）
        forward_returns: 未来收益（index=date）
        method: 'spearman' or 'pearson'
    
    Returns:
        IC值
    """
    aligned = pd.DataFrame({
        'factor': factor_values,
        'return': forward_returns
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    return aligned['factor'].corr(aligned['return'], method=method)


def batch_test_factors(
    df: pd.DataFrame,
    forward_return_column: str = 'forward_20d_return',
    min_ic_threshold: float = 0.05
) -> pd.DataFrame:
    """批量测试所有因子
    
    Returns:
        DataFrame with columns=['factor_name', 'ic', 'category']
        按IC降序排列
    """
    # 收集所有因子
    all_factors = {}
    
    all_factors.update(momentum_factors(df))
    all_factors.update(reversal_factors(df))
    all_factors.update(volatility_factors(df))
    all_factors.update(liquidity_factors(df))
    all_factors.update(worldquant_alpha(df))
    
    # 测试每个因子
    results = []
    forward_returns = df[forward_return_column]
    
    for name, factor_series in all_factors.items():
        ic = test_single_factor_ic(factor_series, forward_returns)
        if not np.isnan(ic) and abs(ic) >= min_ic_threshold:
            category = name.split('_')[0]  # 从名字推断类别
            results.append({
                'factor_name': name,
                'ic': ic,
                'abs_ic': abs(ic),
                'category': category
            })
    
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    
    return result_df.sort_values('abs_ic', ascending=False)


# ==================== VectorBT集成（如果安装）====================

def create_vectorbt_strategy(factor_name: str):
    """创建VectorBT回测策略（需要安装vectorbt）
    
    用法:
        import vectorbt as vbt
        strategy = create_vectorbt_strategy('ret_20d')
        portfolio = vbt.Portfolio.from_signals(
            close, entries, exits, init_cash=100000
        )
        print(portfolio.stats())
    """
    try:
        import vectorbt as vbt
        print(f"VectorBT {vbt.__version__} 已安装，可用于快速回测")
        return True
    except ImportError:
        print("VectorBT未安装。安装命令: pip install vectorbt")
        return False


if __name__ == "__main__":
    # 示例：生成模拟数据并测试因子
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    
    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': 1e6 + np.random.randn(100) * 1e5,
    }, index=dates)
    
    # 计算未来收益
    df['forward_20d_return'] = df['close'].pct_change(20).shift(-20)
    
    # 批量测试因子
    print("测试因子IC...")
    results = batch_test_factors(df, min_ic_threshold=0.0)
    print(results.head(10))
    
    # 检查VectorBT
    create_vectorbt_strategy('test')
