import pandas as pd
import numpy as np

class FinancialMetricsAnalyzer:
    def __init__(self):
        pass
    
    
    def calculate_volatility(self, stock_data):
        """
        Calculate stock volatility
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            
        Returns:
            dict: Volatility metrics by ticker
        """
        volatility_metrics = {}
        
        # Group by ticker
        grouped = stock_data.groupby('symbol')
        
        for symbol, group in grouped:
            # Daily returns
            group = group.sort_values('Date')
            group['daily_return'] = group['Close'].pct_change()
            
            daily_volatility = group['daily_return'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # Annualized
            
            avg_volume = group['Volume'].mean()
            
            # Calculate ATR for recent volatility
            group['high_low'] = group['High'] - group['Low']
            group['high_close'] = abs(group['High'] - group['Close'].shift())
            group['low_close'] = abs(group['Low'] - group['Close'].shift())
            group['true_range'] = group[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = group['true_range'].mean()
            
            volatility_metrics[symbol] = {
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'average_volume': avg_volume,
                'atr': atr,
                'recent_price': group['Close'].iloc[-1]
            }
        
        return volatility_metrics