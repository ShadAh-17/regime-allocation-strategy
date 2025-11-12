"""
Data Loader Module
Downloads and prepares market data for regime analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def download_market_data(tickers=['TLT', 'GLD', 'SPY', '^VIX'], start_date='2004-01-01'):
    """
    Download daily data for ETFs and VIX.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for data download
        
    Returns:
    --------
    pd.DataFrame : Daily adjusted close prices
    """
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)['Adj Close']
    
    # Handle VIX column name
    if '^VIX' in data.columns:
        data = data.rename(columns={'^VIX': 'VIX'})
    
    # Drop missing values
    data = data.dropna()
    
    print(f"Downloaded {len(data)} trading days from {data.index[0].date()} to {data.index[-1].date()}")
    return data

def compute_returns(prices):
    """
    Calculate log returns for ETFs and VIX changes.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Daily prices
        
    Returns:
    --------
    pd.DataFrame : Returns and VIX changes
    """
    returns = pd.DataFrame(index=prices.index)
    
    # Log returns for ETFs
    for col in ['TLT', 'GLD', 'SPY']:
        if col in prices.columns:
            returns[f'{col}_ret'] = np.log(prices[col] / prices[col].shift(1))
    
    # VIX change (not log return since VIX is already volatility)
    if 'VIX' in prices.columns:
        returns['VIX_change'] = prices['VIX'].diff()
        returns['VIX'] = prices['VIX']
    
    # Drop first row (NaN from differencing)
    returns = returns.dropna()
    
    return returns

def load_and_prepare_data(save_path='data/market_data.csv'):
    """
    Main function to download and prepare all data.
    
    Parameters:
    -----------
    save_path : str
        Path to save processed data
        
    Returns:
    --------
    pd.DataFrame : Processed returns data
    """
    # Download prices
    prices = download_market_data()
    
    # Compute returns
    returns = compute_returns(prices)
    
    # Save to CSV
    returns.to_csv(save_path)
    print(f"\nData saved to {save_path}")
    print(f"Shape: {returns.shape}")
    print(f"\nColumns: {list(returns.columns)}")
    
    return returns

if __name__ == "__main__":
    # Test the module
    data = load_and_prepare_data()
    print("\nFirst 5 rows:")
    print(data.head())
