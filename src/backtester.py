"""
Backtesting Module
Performance evaluation and strategy comparison.
"""

import numpy as np
import pandas as pd

class StrategyBacktester:
    """Backtest regime-based allocation strategy."""
    
    def __init__(self, returns_data, states):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            ETF returns
        states : array-like
            Regime states
        """
        self.returns = returns_data
        self.states = states
        self.results = {}
        
    def calculate_regime_performance(self):
        """
        Calculate mean returns by regime for each ETF.
        
        Returns:
        --------
        pd.DataFrame : Mean returns by regime and ETF
        """
        df = pd.DataFrame({
            'state': self.states,
            'TLT_ret': self.returns['TLT_ret'],
            'GLD_ret': self.returns['GLD_ret'],
            'SPY_ret': self.returns['SPY_ret']
        })
        
        # Mean returns by state
        regime_perf = df.groupby('state')[['TLT_ret', 'GLD_ret', 'SPY_ret']].mean()
        regime_perf = regime_perf * 10000  # Convert to basis points
        
        return regime_perf
    
    def create_allocation_rules(self, regime_perf):
        """
        Create allocation rules: pick best ETF for each regime.
        
        Parameters:
        -----------
        regime_perf : pd.DataFrame
            Mean returns by regime
            
        Returns:
        --------
        dict : State -> ETF mapping
        """
        rules = {}
        for state in regime_perf.index:
            best_etf = regime_perf.loc[state].idxmax().replace('_ret', '')
            rules[state] = best_etf
        
        return rules
    
    def backtest_strategy(self, allocation_rules, lag_days=1):
        """
        Backtest the regime-based strategy.
        
        Parameters:
        -----------
        allocation_rules : dict
            State -> ETF mapping
        lag_days : int
            Execution lag (default: 1 day)
            
        Returns:
        --------
        pd.Series : Strategy daily returns
        """
        # Create signals with lag
        signals = pd.Series(self.states, index=self.returns.index)
        signals_lagged = signals.shift(lag_days)
        
        # Map states to allocations
        allocations = signals_lagged.map(allocation_rules)
        
        # Calculate strategy returns
        strategy_returns = pd.Series(index=self.returns.index, dtype=float)
        
        for idx in strategy_returns.index:
            if pd.notna(allocations[idx]):
                etf = allocations[idx]
                strategy_returns[idx] = self.returns.loc[idx, f'{etf}_ret']
        
        return strategy_returns.dropna()
    
    def calculate_metrics(self, returns):
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
            
        Returns:
        --------
        dict : Performance metrics
        """
        # Convert to percent
        returns_pct = returns * 100
        
        # Cumulative return
        cumulative = (1 + returns).prod() - 1
        
        # Annualized metrics
        n_years = len(returns) / 252
        ann_return = (1 + cumulative) ** (1/n_years) - 1
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return': f"{cumulative*100:.1f}%",
            'Annual Return': f"{ann_return*100:.1f}%",
            'Annual Volatility': f"{ann_vol*100:.1f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Sortino Ratio': f"{sortino:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.1f}%",
            'Num Observations': len(returns)
        }
    
    def create_benchmarks(self):
        """
        Create benchmark strategies.
        
        Returns:
        --------
        dict : Benchmark returns
        """
        benchmarks = {}
        
        # Equal weight (rebalanced monthly)
        ew_returns = (self.returns['TLT_ret'] + self.returns['GLD_ret'] + self.returns['SPY_ret']) / 3
        benchmarks['Equal Weight'] = ew_returns
        
        # Buy and hold SPY
        benchmarks['Buy & Hold SPY'] = self.returns['SPY_ret']
        
        return benchmarks
    
    def run_full_backtest(self):
        """
        Run complete backtest with benchmarks.
        
        Returns:
        --------
        pd.DataFrame : Performance comparison table
        """
        # Step 1: Calculate regime performance
        regime_perf = self.calculate_regime_performance()
        print("Regime Performance (basis points per day):")
        print(regime_perf.round(2))
        
        # Step 2: Create allocation rules
        rules = self.create_allocation_rules(regime_perf)
        print("\nAllocation Rules:")
        for state, etf in rules.items():
            print(f"  State {state} -> {etf}")
        
        # Step 3: Backtest strategy
        strategy_returns = self.backtest_strategy(rules)
        
        # Step 4: Calculate metrics
        results = {}
        results['Regime Strategy'] = self.calculate_metrics(strategy_returns)
        
        # Step 5: Benchmark comparison
        benchmarks = self.create_benchmarks()
        for name, returns in benchmarks.items():
            results[name] = self.calculate_metrics(returns)
        
        # Create comparison table
        comparison = pd.DataFrame(results).T
        
        return comparison, strategy_returns

if __name__ == "__main__":
    print("Backtester Module - Ready")
