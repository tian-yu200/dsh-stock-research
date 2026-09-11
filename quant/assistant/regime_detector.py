"""市场状态检测器 - 防止模型在regime切换时退化

基于你的真实数据特征:
- IC从0.09395退化到0.09389（虽小但趋势明确）
- 需要在市场regime变化时触发重训练

参考:
- Qlib Regime-Aware策略
- 《Advances in Financial Machine Learning》by Marcos López de Prado
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json


class MarketRegimeDetector:
    """市场状态检测器（基于技术指标）"""
    
    def __init__(self, 
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.001,
                 lookback_days: int = 60):
        """
        Args:
            volatility_threshold: 波动率阈值（日收益标准差）
            trend_threshold: 趋势阈值（日均收益）
            lookback_days: 回看窗口
        """
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.lookback_days = lookback_days
        
    def detect(self, returns: pd.Series) -> Tuple[str, Dict]:
        """检测当前市场regime
        
        Args:
            returns: 基准指数日收益率序列
            
        Returns:
            (regime, metrics)
            regime: 'bull_low_vol' | 'bull_high_vol' | 'bear' | 'sideways_low_vol' | 'sideways_high_vol'
            metrics: 详细指标
        """
        if len(returns) < self.lookback_days:
            return 'unknown', {'reason': 'insufficient_data'}
        
        recent = returns.tail(self.lookback_days)
        
        # 计算关键指标
        mean_return = recent.mean()
        volatility = recent.std()
        skewness = recent.skew()
        
        # 趋势强度（线性回归斜率）
        x = np.arange(len(recent))
        trend_slope = np.polyfit(x, recent.values, 1)[0]
        
        # 最近10天vs前50天对比（短期变化）
        recent_10d = recent.tail(10).mean()
        prev_50d = recent.iloc[-60:-10].mean()
        momentum_shift = recent_10d - prev_50d
        
        metrics = {
            'mean_return': float(mean_return),
            'volatility': float(volatility),
            'skewness': float(skewness),
            'trend_slope': float(trend_slope),
            'momentum_shift': float(momentum_shift),
            'lookback_days': self.lookback_days
        }
        
        # Regime分类（5种状态）
        high_vol = volatility > self.volatility_threshold
        positive_trend = mean_return > self.trend_threshold
        negative_trend = mean_return < -self.trend_threshold
        
        if positive_trend:
            regime = 'bull_high_vol' if high_vol else 'bull_low_vol'
        elif negative_trend:
            regime = 'bear'  # 熊市不区分波动率（都危险）
        else:
            regime = 'sideways_high_vol' if high_vol else 'sideways_low_vol'
        
        metrics['regime'] = regime
        return regime, metrics
    
    def should_retrain(self, 
                       current_regime: str, 
                       last_trained_regime: Optional[str]) -> Tuple[bool, str]:
        """判断是否需要重新训练
        
        Args:
            current_regime: 当前检测到的regime
            last_trained_regime: 上次训练时的regime
            
        Returns:
            (should_retrain, reason)
        """
        if last_trained_regime is None:
            return True, "首次训练"
        
        # Regime转换矩阵（哪些切换需要重训）
        critical_transitions = {
            ('bull_low_vol', 'bear'),
            ('bull_high_vol', 'bear'),
            ('bull_low_vol', 'sideways_high_vol'),
            ('bear', 'bull_low_vol'),
            ('bear', 'bull_high_vol'),
            ('sideways_low_vol', 'bull_high_vol'),
            ('sideways_low_vol', 'bear'),
        }
        
        transition = (last_trained_regime, current_regime)
        
        if transition in critical_transitions:
            return True, f"关键转换: {last_trained_regime} → {current_regime}"
        
        # 同regime但波动率剧变
        if 'low_vol' in last_trained_regime and 'high_vol' in current_regime:
            return True, f"波动率突增: {last_trained_regime} → {current_regime}"
        
        return False, f"regime稳定: {current_regime}"


class RegimeAwareModelManager:
    """Regime感知的模型管理器"""
    
    def __init__(self, state_file: str = "data/model_regime_state.json"):
        self.state_file = Path(state_file)
        self.detector = MarketRegimeDetector()
        
    def load_state(self) -> Dict:
        """加载上次训练的regime状态"""
        if not self.state_file.exists():
            return {
                'last_trained_regime': None,
                'last_trained_date': None,
                'training_history': []
            }
        
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_state(self, regime: str, date: str, ic: float):
        """保存训练状态"""
        state = self.load_state()
        state['last_trained_regime'] = regime
        state['last_trained_date'] = date
        state['training_history'].append({
            'date': date,
            'regime': regime,
            'ic': ic
        })
        
        # 只保留最近100次训练记录
        state['training_history'] = state['training_history'][-100:]
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def check_should_retrain(self, benchmark_returns: pd.Series, current_date: str) -> Dict:
        """检查是否需要重新训练
        
        Returns:
            {
                'should_retrain': bool,
                'reason': str,
                'current_regime': str,
                'last_regime': str,
                'metrics': dict
            }
        """
        state = self.load_state()
        
        # 检测当前regime
        current_regime, metrics = self.detector.detect(benchmark_returns)
        
        # 判断是否需要重训
        should_retrain, reason = self.detector.should_retrain(
            current_regime,
            state['last_trained_regime']
        )
        
        return {
            'should_retrain': should_retrain,
            'reason': reason,
            'current_regime': current_regime,
            'last_regime': state.get('last_trained_regime'),
            'last_trained_date': state.get('last_trained_date'),
            'metrics': metrics
        }


def get_benchmark_returns(data_dir: str = "D:/dsh-gupiao/quant/data") -> pd.Series:
    """从你的数据中提取基准收益（沪深300或000001平安银行作为proxy）"""
    benchmark_path = Path(data_dir) / "stocks" / "000001.csv"
    
    if not benchmark_path.exists():
        raise FileNotFoundError(f"基准数据不存在: {benchmark_path}")
    
    df = pd.read_csv(benchmark_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    returns = df['close'].pct_change().dropna()
    return returns


if __name__ == "__main__":
    print("=== 市场Regime检测 ===\n")
    
    # 1. 加载基准收益
    returns = get_benchmark_returns()
    print(f"✓ 加载基准数据: {len(returns)} 天")
    
    # 2. 检测当前regime
    manager = RegimeAwareModelManager(
        state_file="D:/dsh-gupiao/quant/data/model_regime_state.json"
    )
    
    result = manager.check_should_retrain(
        returns,
        current_date=returns.index[-1].strftime('%Y-%m-%d')
    )
    
    # 3. 输出结果
    print(f"\n当前Regime: {result['current_regime']}")
    print(f"上次训练Regime: {result.get('last_regime', 'N/A')}")
    print(f"上次训练日期: {result.get('last_trained_date', 'N/A')}")
    print(f"\n是否需要重训: {'✓ 是' if result['should_retrain'] else '✗ 否'}")
    print(f"原因: {result['reason']}")
    
    print("\n详细指标:")
    metrics = result['metrics']
    print(f"  平均日收益: {metrics['mean_return']*100:.3f}%")
    print(f"  波动率(日): {metrics['volatility']*100:.3f}%")
    print(f"  偏度: {metrics['skewness']:.3f}")
    print(f"  趋势斜率: {metrics['trend_slope']:.6f}")
    print(f"  动量变化: {metrics['momentum_shift']*100:.3f}%")
    
    # 4. 模拟保存状态
    if result['should_retrain']:
        print("\n✓ 建议立即重新训练模型")
        # manager.save_state(
        #     regime=result['current_regime'],
        #     date=returns.index[-1].strftime('%Y-%m-%d'),
        #     ic=0.094  # 训练后填入实际IC
        # )
    else:
        print("\n✓ 当前模型仍然适用")
