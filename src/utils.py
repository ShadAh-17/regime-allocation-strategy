"""
Utility functions for visualization and reporting.
Author: Shadaab Ahmed
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def plot_regime_classification(dates, vix, states, title="VIX Regime Classification"):
    """
    Plot VIX with color-coded regimes.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Date index
    vix : array-like
        VIX values
    states : array-like
        Regime states
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create colormap
    unique_states = sorted(set(states))
    colors = ['green', 'orange', 'red'][:len(unique_states)]
    
    for state, color in zip(unique_states, colors):
        mask = states == state
        ax.scatter(dates[mask], vix[mask], c=color, s=1, alpha=0.6, 
                  label=f'State {state}')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('VIX Level')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_cumulative_returns(returns_dict, title="Cumulative Returns Comparison"):
    """
    Plot cumulative returns for multiple strategies.
    
    Parameters:
    -----------
    returns_dict : dict
        Dictionary of {strategy_name: returns_series}
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (1 = 100%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add crisis shading
    crisis_periods = [
        ('2008-09-01', '2009-03-31', '2008 Crisis'),
        ('2020-02-01', '2020-04-30', 'COVID-19'),
    ]
    
    for start, end, label in crisis_periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                   alpha=0.1, color='red')
    
    plt.tight_layout()
    return fig

def plot_drawdowns(returns_dict, title="Drawdown Comparison"):
    """
    Plot drawdowns for multiple strategies.
    
    Parameters:
    -----------
    returns_dict : dict
        Dictionary of {strategy_name: returns_series}
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_performance_bars(comparison_df, metrics=['Total Return', 'Sharpe Ratio', 'Max Drawdown']):
    """
    Bar chart comparing key metrics.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Performance comparison table
    metrics : list
        Metrics to plot
    """
    # Extract numeric values
    plot_data = pd.DataFrame()
    for metric in metrics:
        values = comparison_df[metric].str.rstrip('%').astype(float)
        plot_data[metric] = values
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx] if len(metrics) > 1 else axes
        plot_data[metric].plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def generate_summary_table(comparison_df, save_path=None):
    """
    Generate formatted performance summary.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Performance metrics
    save_path : str
        Path to save table (optional)
    """
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(comparison_df.to_string())
    print("="*70)
    
    if save_path:
        comparison_df.to_csv(save_path)
        print(f"\nResults saved to: {save_path}")
    
    return comparison_df

if __name__ == "__main__":
    print("Visualization Utils - Ready")
