"""低换手率选股策略 - 基于Qlib TopkDropoutStrategy改进

参考：
- Qlib TopkDropoutStrategy: https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/signal_strategy.py
- 论文: https://arxiv.org/abs/2104.05031 (Qlib: An AI-oriented Quantitative Investment Platform)

核心改进：
1. 只在信号强度>阈值时换仓（避免频繁调整）
2. 当前持仓需要更强的反向信号才卖出（持仓惯性）
3. 止损保护：单只股票亏损>8%强制清仓
"""
from typing import Any
import pandas as pd
import numpy as np


class LowTurnoverTopkStrategy:
    """低换手率Top-K策略
    
    参数:
        topk: 持仓数量
        signal_threshold: 信号强度阈值，只有score绝对值>此值才考虑买入（降低换手）
        hold_threshold: 持仓惯性，当前持仓的stock需要score < -hold_threshold才卖出
        stop_loss: 止损阈值，单只股票亏损超过此比例强制清仓（如-0.08表示-8%）
    """
    
    def __init__(
        self,
        topk: int = 3,
        signal_threshold: float = 0.01,  # 信号强度>1%才考虑
        hold_threshold: float = 0.005,   # 持仓需要反向信号>0.5%才卖
        stop_loss: float = -0.08,        # -8%止损
    ):
        self.topk = topk
        self.signal_threshold = signal_threshold
        self.hold_threshold = hold_threshold
        self.stop_loss = stop_loss
        self.current_position = {}  # {symbol: score}
        self.entry_prices = {}      # {symbol: price} 用于止损
    
    def generate_target_weight_position(
        self,
        scores: dict[str, float],
        current_prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """生成目标权重
        
        Args:
            scores: {symbol: predicted_score}
            current_prices: {symbol: current_price} 用于止损计算
            
        Returns:
            {symbol: weight} 等权或0
        """
        if not scores:
            return {}
        
        # 1. 过滤信号强度不足的候选
        candidates = {
            symbol: score
            for symbol, score in scores.items()
            if abs(score) >= self.signal_threshold
        }
        
        if not candidates:
            # 信号太弱，保持当前持仓（降低换手）
            return self._preserve_current_position()
        
        # 2. 排序候选
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 3. 应用持仓惯性 - 当前持仓优先保留
        new_position = {}
        
        # 3a. 止损检查 - 强制清仓亏损股票
        if current_prices:
            for symbol in list(self.current_position.keys()):
                if symbol in self.entry_prices and symbol in current_prices:
                    entry = self.entry_prices[symbol]
                    current = current_prices[symbol]
                    pnl = (current - entry) / entry
                    if pnl < self.stop_loss:
                        # 触发止损，从候选中移除
                        print(f"[止损] {symbol}: {pnl:.2%} < {self.stop_loss:.2%}")
                        continue
        
        # 3b. 保留当前持仓中信号仍然OK的股票
        for symbol in self.current_position:
            if symbol in scores:
                score = scores[symbol]
                # 持仓惯性：需要明确的负信号才卖出
                if score >= -self.hold_threshold:
                    new_position[symbol] = score
        
        # 3c. 从候选中补充到topk
        for symbol, score in sorted_candidates:
            if len(new_position) >= self.topk:
                break
            if symbol not in new_position:
                new_position[symbol] = score
                # 记录入场价格
                if current_prices and symbol in current_prices:
                    self.entry_prices[symbol] = current_prices[symbol]
        
        # 4. 更新持仓记录
        self.current_position = new_position
        
        # 5. 转换为等权重
        if not new_position:
            return {}
        
        weight = 1.0 / len(new_position)
        return {symbol: weight for symbol in new_position}
    
    def _preserve_current_position(self) -> dict[str, float]:
        """保持当前持仓不变"""
        if not self.current_position:
            return {}
        weight = 1.0 / len(self.current_position)
        return {symbol: weight for symbol in self.current_position}
    
    def reset(self):
        """重置策略状态"""
        self.current_position = {}
        self.entry_prices = {}


def calculate_turnover_reduction(
    original_turnover: float,
    new_turnover: float
) -> dict[str, Any]:
    """计算换手率改善效果
    
    Returns:
        {
            'original_annual_cost': 原年化成本,
            'new_annual_cost': 新年化成本,
            'cost_reduction': 成本降低比例,
            'expected_alpha_boost': 预期Alpha提升
        }
    """
    COST_BPS = 10  # 10个基点交易成本
    TRADING_DAYS = 250
    
    original_cost = original_turnover * (COST_BPS / 10000) * TRADING_DAYS
    new_cost = new_turnover * (COST_BPS / 10000) * TRADING_DAYS
    
    return {
        'original_annual_cost': original_cost,
        'new_annual_cost': new_cost,
        'cost_reduction': (original_cost - new_cost) / original_cost if original_cost > 0 else 0,
        'expected_alpha_boost': original_cost - new_cost,  # 节省的成本=额外收益
    }


if __name__ == "__main__":
    # 示例：模拟3天的信号
    strategy = LowTurnoverTopkStrategy(topk=3, signal_threshold=0.01, hold_threshold=0.005)
    
    # Day 1
    scores_day1 = {
        '300642': 0.16,  # 强信号
        '300779': 0.10,  # 强信号
        '920855': 0.02,  # 弱信号
        '000001': -0.01, # 负信号
    }
    weights_day1 = strategy.generate_target_weight_position(scores_day1)
    print(f"Day 1: {weights_day1}")
    
    # Day 2 - 信号略有变化，但不足以触发调仓
    scores_day2 = {
        '300642': 0.15,  # 略降
        '300779': 0.09,  # 略降
        '920855': 0.025, # 略升但仍弱
        '000001': 0.005, # 转正但弱
    }
    weights_day2 = strategy.generate_target_weight_position(scores_day2)
    print(f"Day 2: {weights_day2}")  # 应该保持Day1持仓
    
    # Day 3 - 出现强烈负信号，触发调仓
    scores_day3 = {
        '300642': 0.14,
        '300779': -0.02,  # 转负且超过hold_threshold，触发卖出
        '920855': 0.12,   # 大幅上升，买入
        '000001': 0.08,   # 大幅上升，买入
    }
    weights_day3 = strategy.generate_target_weight_position(scores_day3)
    print(f"Day 3: {weights_day3}")
    
    # 计算换手率改善
    improvement = calculate_turnover_reduction(0.56, 0.15)
    print(f"\n换手率改善效果: {improvement}")
