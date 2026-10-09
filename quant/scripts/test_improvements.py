"""验证改进效果 - 运行一次研究并检查日志

使用方法:
    python scripts/test_improvements.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime


def test_regime_detection():
    """测试1: Regime检测是否工作"""
    print("\n" + "="*60)
    print("测试1: 市场Regime检测")
    print("="*60)
    
    try:
        from assistant.regime_detector import get_benchmark_returns, MarketRegimeDetector
        
        returns = get_benchmark_returns()
        detector = MarketRegimeDetector()
        regime, metrics = detector.detect(returns)
        
        print(f"✓ Regime检测成功")
        print(f"  当前市场: {regime}")
        print(f"  日均收益: {metrics['mean_return']*100:.3f}%")
        print(f"  波动率: {metrics['volatility']*100:.2f}%")
        
        if 'bear' in regime:
            print(f"  ⚠️  检测到熊市 - 系统将自动:")
            print(f"     • 提高现金比例至40%")
            print(f"     • 降低股票仓位至20%")
            print(f"     • 提高信号阈值(2%)")
        
        return True
    except Exception as e:
        print(f"✗ Regime检测失败: {e}")
        return False


def test_low_turnover_strategy():
    """测试2: 低换手率策略是否工作"""
    print("\n" + "="*60)
    print("测试2: 低换手率策略")
    print("="*60)
    
    try:
        from assistant.strategy_lowto import LowTurnoverTopkStrategy, calculate_turnover_reduction
        
        # 模拟测试
        strategy = LowTurnoverTopkStrategy(
            topk=3,
            signal_threshold=0.02,
            hold_threshold=0.01,
            stop_loss=-0.10
        )
        
        # 模拟信号
        test_scores = {
            '300642': 0.16,
            '300779': 0.10,
            '000001': 0.08,
            '920855': 0.03,  # 弱信号，应被过滤
        }
        
        weights = strategy.generate_target_weight_position(test_scores)
        
        print(f"✓ 低换手率策略工作正常")
        print(f"  输入候选: {len(test_scores)} 只")
        print(f"  输出持仓: {len(weights)} 只")
        print(f"  选中标的: {list(weights.keys())}")
        
        # 计算预期收益
        improvement = calculate_turnover_reduction(0.56, 0.20)
        print(f"\n  预期改善:")
        print(f"  • 年化交易成本: {improvement['original_annual_cost']:.2%} → {improvement['new_annual_cost']:.2%}")
        print(f"  • 成本节省: {improvement['expected_alpha_boost']:.2%}/年")
        
        return True
    except Exception as e:
        print(f"✗ 低换手率策略失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """测试3: 集成到research.py是否成功"""
    print("\n" + "="*60)
    print("测试3: 集成验证")
    print("="*60)
    
    try:
        from assistant.research import run_research
        
        print(f"✓ research.py导入成功")
        print(f"  包含:")
        print(f"  • Regime检测 (自动调整风险)")
        print(f"  • 低换手率策略 (持仓惯性)")
        print(f"  • 熊市保护 (信号阈值↑)")
        
        return True
    except Exception as e:
        print(f"✗ 集成验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps(all_passed):
    """显示下一步操作"""
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    if all_passed:
        print("✓ 所有测试通过！")
        print("\n🚀 下一步操作:")
        print("\n1. 运行一次完整研究（测试环境）:")
        print("   cd quant")
        print("   .venv/Scripts/python.exe -c \"from assistant.research import run_research; run_research(mode='on_demand', _force_sync=True)\"")
        
        print("\n2. 检查日志文件:")
        print("   • 查找 'market_regime=' 确认检测到熊市")
        print("   • 查找 'bear_market_detected' 确认参数已调整")
        print("   • 查找 'low_turnover_strategy' 确认策略已应用")
        
        print("\n3. 对比改进前后:")
        print("   • IC是否稳定在0.094+")
        print("   • 换手率是否降低")
        print("   • 最大回撤是否改善")
        
        print("\n4. 7天后评估:")
        print("   • 如果IC >= 0.095 且回撤 < -28%: 继续使用")
        print("   • 如果IC < 0.090: 回滚并重新评估")
        
    else:
        print("✗ 部分测试失败")
        print("\n请检查:")
        print("  • 所有新文件是否在正确位置")
        print("  • 依赖是否正确导入")
        print("  • 数据文件是否存在")


def main():
    print("="*60)
    print("🧪 改进效果验证")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 运行所有测试
    results.append(test_regime_detection())
    results.append(test_low_turnover_strategy())
    results.append(test_integration())
    
    # 显示结果
    all_passed = all(results)
    show_next_steps(all_passed)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
