"""
Regime Detection Module
Identifies market volatility regimes using HMM.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class RegimeDetector:
    """Detect market regimes using Hidden Markov Model."""
    
    def __init__(self, n_regimes=3):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        n_regimes : int
            Number of volatility regimes (default: 3)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, vix_changes, random_state=42):
        """
        Fit HMM to VIX changes.
        
        Parameters:
        -----------
        vix_changes : array-like
            Daily VIX changes
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        self
        """
        # Reshape and standardize
        X = np.array(vix_changes).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=random_state
        )
        
        self.model.fit(X_scaled)
        print(f"Fitted {self.n_regimes}-state HMM")
        print(f"Converged: {self.model.monitor_.converged}")
        
        return self
    
    def predict(self, vix_changes):
        """
        Predict regime states.
        
        Parameters:
        -----------
        vix_changes : array-like
            Daily VIX changes
            
        Returns:
        --------
        np.array : Predicted states
        """
        X = np.array(vix_changes).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        return states
    
    def get_regime_stats(self, states, vix_changes):
        """
        Calculate statistics for each regime.
        
        Parameters:
        -----------
        states : array-like
            Regime states
        vix_changes : array-like
            VIX changes
            
        Returns:
        --------
        pd.DataFrame : Mean and std of VIX changes by regime
        """
        df = pd.DataFrame({
            'state': states,
            'vix_change': vix_changes
        })
        
        stats = df.groupby('state')['vix_change'].agg(['mean', 'std', 'count'])
        stats = stats.sort_values('mean')
        
        # Rename states based on volatility level
        state_names = {
            stats.index[0]: 'Low Vol',
            stats.index[1]: 'Medium Vol',
            stats.index[2]: 'High Vol'
        } if self.n_regimes == 3 else {
            stats.index[0]: 'Low Vol',
            stats.index[1]: 'High Vol'
        }
        
        stats['regime_name'] = stats.index.map(state_names)
        
        return stats
    
    def score_model(self, vix_changes):
        """
        Calculate model selection criteria.
        
        Parameters:
        -----------
        vix_changes : array-like
            VIX changes
            
        Returns:
        --------
        dict : AIC and BIC scores
        """
        X = np.array(vix_changes).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        
        log_likelihood = self.model.score(X_scaled)
        n_params = self.n_regimes ** 2 + 2 * self.n_regimes  # Transition matrix + means + variances
        n_samples = len(vix_changes)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        return {
            'n_states': self.n_regimes,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic
        }

def compare_models(vix_changes, n_states_list=[2, 3, 4]):
    """
    Compare HMM models with different number of states.
    
    Parameters:
    -----------
    vix_changes : array-like
        VIX changes
    n_states_list : list
        List of state counts to compare
        
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    results = []
    
    for n in n_states_list:
        detector = RegimeDetector(n_regimes=n)
        detector.fit(vix_changes)
        scores = detector.score_model(vix_changes)
        results.append(scores)
    
    comparison = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(comparison)
    
    # Highlight best model
    best_aic = comparison.loc[comparison['AIC'].idxmin(), 'n_states']
    best_bic = comparison.loc[comparison['BIC'].idxmin(), 'n_states']
    print(f"\nBest by AIC: {best_aic} states")
    print(f"Best by BIC: {best_bic} states")
    
    return comparison

if __name__ == "__main__":
    # Test the module
    print("Regime Detector Module - Ready")
