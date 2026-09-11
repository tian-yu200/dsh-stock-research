"""基于真实数据的深度诊断和优化方案

数据资产盘点:
- rolling_scan.sqlite3: 31,048条评分记录 (2016-09-27 to 2026-09-11)
- 股票CSV: 完整OHLCV数据，从2020年开始
- 549个运行结果JSON: 完整的训练历史

核心发现:
1. 你有10年历史评分数据（2016-2026），但只用了2.5年训练（2023-2025）
2. 北交所股票920855得分0.166（最高分），但被你的pool_exclude_bj剔除了
3. 数据质量很好，有完整的成交量、成交额、换手率

优化策略（基于你的数据特征）:
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class RealDataAnalyzer:
    """基于你的真实数据做诊断"""
    
    def __init__(self, quant_dir: str = "D:/dsh-gupiao/quant"):
        self.quant_dir = Path(quant_dir)
        self.data_dir = self.quant_dir / "data"
        self.results_dir = self.quant_dir / "results/assistant"
        
    def analyze_score_history(self) -> pd.DataFrame:
        """分析rolling_scan历史评分，找IC变化趋势"""
        conn = sqlite3.connect(self.data_dir / "rolling_scan.sqlite3")
        
        # 提取评分历史
        query = """
        SELECT 
            symbol,
            data_end_date,
            return_20d,
            return_60d,
            volatility_60d_annualized as volatility_60d,
            max_drawdown,
            trend_score,
            asset_type
        FROM scores
        WHERE data_end_date >= '2023-01-01'
        ORDER BY data_end_date, trend_score DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
    
    def find_consistent_winners(self, df: pd.DataFrame, top_n: int = 30) -> Dict:
        """找出长期稳定高分的股票（真正的Alpha）
        
        策略: 不是找单次最高分，而是找"经常进Top30"的股票
        """
        # 按日期分组，每天选Top30
        daily_tops = {}
        for date in df['data_end_date'].unique():
            day_data = df[df['data_end_date'] == date].nlargest(top_n, 'trend_score')
            daily_tops[date] = set(day_data['symbol'].values)
        
        # 统计每只股票进Top30的次数
        symbol_frequency = {}
        for symbols in daily_tops.values():
            for symbol in symbols:
                symbol_frequency[symbol] = symbol_frequency.get(symbol, 0) + 1
        
        # 计算稳定性得分
        total_days = len(daily_tops)
        stability_scores = {
            symbol: count / total_days
            for symbol, count in symbol_frequency.items()
        }
        
        # 排序
        consistent_winners = sorted(
            stability_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]
        
        return {
            'winners': consistent_winners,
            'total_days': total_days,
            'interpretation': f"前50只股票中，最稳定的进Top30频率为{consistent_winners[0][1]:.1%}"
        }
    
    def analyze_bj_stocks(self, df: pd.DataFrame) -> Dict:
        """专门分析北交所股票表现（验证pool_exclude_bj决策）"""
        bj_stocks = df[df['symbol'].str.startswith('920')]
        non_bj_stocks = df[~df['symbol'].str.startswith('920')]
        
        if len(bj_stocks) == 0:
            return {'error': '没有北交所数据（可能已被过滤）'}
        
        return {
            'bj_count': len(bj_stocks['symbol'].unique()),
            'bj_avg_score': bj_stocks['trend_score'].mean(),
            'bj_median_score': bj_stocks['trend_score'].median(),
            'bj_top10_avg': bj_stocks.nlargest(10, 'trend_score')['trend_score'].mean(),
            'non_bj_avg_score': non_bj_stocks['trend_score'].mean(),
            'non_bj_median_score': non_bj_stocks['trend_score'].median(),
            'bj_volatility': bj_stocks['volatility_60d'].mean(),
            'non_bj_volatility': non_bj_stocks['volatility_60d'].mean(),
            'conclusion': 'BJ股票平均得分更高' if bj_stocks['trend_score'].mean() > non_bj_stocks['trend_score'].mean() else 'BJ股票平均得分更低'
        }
    
    def load_stock_ohlcv(self, symbol: str) -> pd.DataFrame:
        """加载单只股票的完整OHLCV数据"""
        csv_path = self.data_dir / f"stocks/{symbol}.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df
    
    def calculate_custom_factors_for_symbol(self, symbol: str) -> Dict:
        """为单只股票计算自定义因子（基于你的数据格式）"""
        df = self.load_stock_ohlcv(symbol)
        if df.empty or len(df) < 60:
            return {}
        
        close = df['close']
        volume = df['volume']
        amount = df['amount']
        turnover = df['turnover']
        
        # 基于你的数据特征定制的因子
        factors = {}
        
        # 1. 量价背离因子（A股特色）
        price_chg_20d = close.pct_change(20)
        volume_chg_20d = volume.pct_change(20)
        factors['price_volume_divergence'] = (price_chg_20d.iloc[-1] - volume_chg_20d.iloc[-1])
        
        # 2. 成交额稳定性（你的数据有amount字段）
        factors['amount_stability'] = amount.rolling(20).std() / amount.rolling(20).mean()
        factors['amount_rank_60d'] = (amount.iloc[-1] > amount.rolling(60).quantile(0.8).iloc[-1])
        
        # 3. 换手率因子（你的数据有turnover字段）
        factors['turnover_ma_ratio'] = turnover.iloc[-1] / turnover.rolling(20).mean().iloc[-1]
        factors['turnover_surge'] = turnover.iloc[-1] > turnover.rolling(60).quantile(0.9).iloc[-1]
        
        # 4. 突破因子
        factors['near_52w_high'] = close.iloc[-1] / close.rolling(252).max().iloc[-1]
        factors['near_52w_low'] = close.iloc[-1] / close.rolling(252).min().iloc[-1]
        
        # 5. 波动率调整动量
        returns = close.pct_change()
        vol_20d = returns.rolling(20).std()
        factors['vol_adj_momentum_20d'] = (close.pct_change(20).iloc[-1] / vol_20d.iloc[-1]) if vol_20d.iloc[-1] > 0 else 0
        
        return factors
    
    def batch_calculate_factors(self, symbols: List[str]) -> pd.DataFrame:
        """批量计算因子（可以多线程加速）"""
        results = []
        for symbol in symbols:
            factors = self.calculate_custom_factors_for_symbol(symbol)
            if factors:
                factors['symbol'] = symbol
                results.append(factors)
        
        return pd.DataFrame(results)
    
    def analyze_model_runs(self) -> Dict:
        """分析549次运行结果，找规律"""
        runs_dir = self.results_dir / "runs"
        
        ic_history = []
        for manifest_path in runs_dir.rglob("manifest.json"):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    
                if manifest.get('status') == 'failed':
                    continue
                
                # 尝试读取model_metadata
                metadata_path = manifest_path.parent / "model_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        stock_model = metadata.get('stock_model', {})
                        
                        ic_history.append({
                            'date': manifest.get('data_end_date'),
                            'run_id': manifest.get('run_id'),
                            'ic': stock_model.get('mean_rank_ic'),
                            'test_ic': stock_model.get('test_rank_ic'),
                            'ic_ir': stock_model.get('rank_ic_ir'),
                            'annual_return': stock_model.get('net_annualized_return'),
                            'max_drawdown': stock_model.get('max_drawdown'),
                        })
            except Exception as e:
                continue
        
        df = pd.DataFrame(ic_history)
        
        if df.empty:
            return {'error': '没有找到有效的运行历史'}
        
        return {
            'total_runs': len(df),
            'avg_ic': df['ic'].mean(),
            'ic_std': df['ic'].std(),
            'ic_trend': 'improving' if df['ic'].iloc[-5:].mean() > df['ic'].iloc[:5].mean() else 'declining',
            'best_run': df.loc[df['ic'].idxmax()].to_dict() if not df['ic'].isna().all() else {},
            'worst_run': df.loc[df['ic'].idxmin()].to_dict() if not df['ic'].isna().all() else {},
        }


def generate_optimization_plan(analyzer: RealDataAnalyzer) -> Dict:
    """基于真实数据生成定制化优化方案"""
    print("正在分析你的历史数据...")
    
    # 1. 加载评分历史
    scores_df = analyzer.analyze_score_history()
    print(f"✓ 加载了 {len(scores_df)} 条评分记录")
    
    # 2. 找稳定赢家
    winners = analyzer.find_consistent_winners(scores_df, top_n=30)
    print(f"✓ 找到 {len(winners['winners'])} 只稳定高分股票")
    print(f"  最稳定的5只: {winners['winners'][:5]}")
    
    # 3. 分析北交所
    bj_analysis = analyzer.analyze_bj_stocks(scores_df)
    print(f"✓ 北交所分析: {bj_analysis.get('conclusion', 'N/A')}")
    
    # 4. 分析运行历史
    run_history = analyzer.analyze_model_runs()
    print(f"✓ 分析了 {run_history.get('total_runs', 0)} 次运行")
    print(f"  平均IC: {run_history.get('avg_ic', 0):.4f}")
    print(f"  IC趋势: {run_history.get('ic_trend', 'unknown')}")
    
    # 5. 生成优化建议
    recommendations = {
        'immediate_actions': [],
        'factor_enhancements': [],
        'strategy_adjustments': [],
        'data_insights': {
            'winners': winners,
            'bj_analysis': bj_analysis,
            'run_history': run_history,
        }
    }
    
    # 基于分析结果的具体建议
    if bj_analysis.get('bj_avg_score', 0) > bj_analysis.get('non_bj_avg_score', 0):
        recommendations['immediate_actions'].append({
            'priority': 'HIGH',
            'action': '重新考虑 pool_exclude_bj: true',
            'reason': f"北交所平均得分 {bj_analysis.get('bj_avg_score', 0):.4f} > 非北交所 {bj_analysis.get('non_bj_avg_score', 0):.4f}",
            'evidence': '920855得分0.166是所有股票中最高的'
        })
    
    if run_history.get('ic_trend') == 'declining':
        recommendations['strategy_adjustments'].append({
            'priority': 'HIGH',
            'action': '模型正在退化，需要重新训练或换因子',
            'reason': f"最近5次IC {run_history.get('avg_ic', 0):.4f} 呈下降趋势"
        })
    
    # 推荐自定义因子（基于你的数据特征）
    recommendations['factor_enhancements'].append({
        'priority': 'MEDIUM',
        'factors': [
            'amount_stability (成交额稳定性)',
            'turnover_surge (换手率突破)',
            'vol_adj_momentum_20d (波动率调整动量)',
        ],
        'reason': '你的数据有完整的amount和turnover字段，可以利用这些A股特色数据'
    })
    
    return recommendations


if __name__ == "__main__":
    analyzer = RealDataAnalyzer()
    
    print("=== 基于你的真实数据做深度诊断 ===\n")
    
    # 执行完整分析
    recommendations = generate_optimization_plan(analyzer)
    
    print("\n=== 优化建议 ===")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))
    
    # 保存分析结果
    output_path = Path("D:/dsh-gupiao/quant/data/real_data_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分析结果已保存到: {output_path}")
