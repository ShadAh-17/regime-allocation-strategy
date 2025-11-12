# Regime-Based Asset Allocation Strategy

**A systematic approach to tactical allocation using volatility regimes**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements a quantitative trading strategy that dynamically allocates between stocks (SPY), bonds (TLT), and gold (GLD) based on market volatility regimes detected using Hidden Markov Models.

**Key Achievement:** The strategy delivered substantial outperformance vs buy-and-hold while significantly reducing drawdowns during crisis periods.

## Project Structure

```
regime-allocation-strategy/
│
├── src/                          # Core Python modules
│   ├── data_loader.py           # Data download and preparation
│   ├── regime_detector.py       # HMM regime identification
│   ├── backtester.py            # Strategy backtesting engine
│   └── utils.py                 # Visualization utilities
│
├── notebooks/                    # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_regime_analysis.ipynb
│   └── 03_strategy_backtest.ipynb
│
├── data/                        # Market data (generated)
├── results/                     # Performance metrics & charts
├── docs/                        # Documentation
└── README.md
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/regime-allocation-strategy.git
cd regime-allocation-strategy

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Execute notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_regime_analysis.ipynb
jupyter notebook notebooks/03_strategy_backtest.ipynb
```

## Methodology

### 1. Data Preparation
- Download 20+ years of daily data for TLT, GLD, SPY, and VIX
- Calculate log returns and VIX changes
- Clean and align time series

### 2. Regime Detection
- Fit Hidden Markov Models with 2, 3, and 4 states
- Select optimal model using AIC/BIC criteria
- **Result:** 3-state model identifies Low/Medium/High volatility regimes

### 3. Allocation Rules
Analyze asset performance by regime and create allocation rules:
- **Low/Medium Volatility** → SPY (stocks outperform in calm markets)
- **High Volatility** → TLT (bonds provide safety during crises)

### 4. Backtesting
- Apply 1-day execution lag (avoid lookahead bias)
- Compare against two benchmarks:
  - Equal-weight allocation (33% each asset)
  - Buy-and-hold SPY

## Results

### Performance Metrics

| Metric | Regime Strategy | Buy & Hold SPY | Equal Weight |
|--------|----------------|----------------|--------------|
| **Total Return** | Excellent | Good | Moderate |
| **Sharpe Ratio** | > 1.0 | ~0.5 | ~0.8 |
| **Max Drawdown** | Low | High | Moderate |
| **Crisis Protection** | Strong | Weak | Moderate |

*Note: Specific numbers available in `results/performance_summary.csv`*

### Key Insights

✅ **Superior Risk-Adjusted Returns:** Significantly higher Sharpe ratio vs benchmarks  
✅ **Downside Protection:** Avoided worst losses during 2008 and 2020 crises  
✅ **Low Turnover:** ~7 rebalances per year (practical for implementation)  
✅ **Simple Rules:** Easy to explain and replicate  

## Visualizations

All charts saved in `results/` folder:
- `cumulative_returns.png` - Performance comparison over time
- `drawdown_analysis.png` - Risk management during crises
- `regime_classification.png` - VIX regimes color-coded
- `risk_return_scatter.png` - Efficient frontier comparison


## Business Applications

**Portfolio Management:**
- Systematic risk management without sacrificing returns
- Clear decision rules for allocation changes

**Risk Management:**
- Early detection of regime shifts
- Automatic defensive positioning

**Institutional Use:**
- Transparent, rules-based process
- Low turnover reduces costs
- Scalable to large AUM

## Limitations & Future Work

**Current Limitations:**
1. In-sample testing only (no out-of-sample validation)
2. Transaction costs not included
3. Limited to three asset classes
4. Assumes historical regimes persist

**Potential Enhancements:**
1. Walk-forward optimization
2. Transaction cost analysis
3. Additional asset classes (international, commodities)
4. Machine learning alternatives (LSTM, Random Forest)
5. Combine with momentum/value factors


## License

MIT License - See LICENSE file for details

## References

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates"
- HMM Documentation: https://hmmlearn.readthedocs.io/

---
