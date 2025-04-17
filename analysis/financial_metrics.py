import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from config.config import (
    HIGH_SHORT_INTEREST,
    HIGH_DAYS_TO_COVER,
    MIN_FLOAT_PERCENTAGE
)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    def calculate_short_squeeze_metrics(self, short_interest_data, stock_data, options_data=None):
        """
        Short squeeze metrics calculation
        
        Args:
            short_interest_data (pandas.DataFrame): Short interest data
            stock_data (pandas.DataFrame): Historical stock data
            options_data (pandas.DataFrame): Options data
            
        Returns:
            pandas.DataFrame: Short squeeze metrics
        """

        squeeze_metrics = []
        
        # Get the most recent date for stock data
        stock_latest = stock_data.groupby('symbol')['Date'].max().reset_index()
        stock_latest_df = pd.merge(
            stock_data, 
            stock_latest, 
            on=['symbol', 'Date']
        )
        
        # Prepare options data dict and calculate metrics for tickers
        options_dict = {}
        if options_data is not None and not options_data.empty:
            for _, row in options_data.iterrows():
                options_dict[row['symbol']] = row.to_dict()
        
        logger.info("Calculating short squeeze metrics...")
        
        for _, row in tqdm(short_interest_data.iterrows(), total=len(short_interest_data)):
            symbol = row['symbol']
            
            # Skip if missing key data
            if pd.isna(row['short_percent_of_float']) or pd.isna(row['short_ratio']):
                continue
                
            # Get latest stock data
            latest_stock = stock_latest_df[stock_latest_df['symbol'] == symbol]
            if latest_stock.empty:
                continue
                
            # Get stock price and volume, and calculate days to cover
            price = latest_stock['Close'].values[0]
            volume = latest_stock['Volume'].values[0]
            
            days_to_cover = row['short_ratio']
            if pd.isna(days_to_cover) and not pd.isna(row['short_interest']) and volume > 0:
                days_to_cover = row['short_interest'] / volume
            
            # Calculate short interest as percentage of float and tital market value of short positions
            si_percent = row['short_percent_of_float']
            short_value = None
            if not pd.isna(row['short_interest']):
                short_value = row['short_interest'] * price
            
            # Options data
            call_volume = None
            put_call_ratio = None
            if symbol in options_dict:
                call_volume = options_dict[symbol].get('call_volume')
                put_volume = options_dict[symbol].get('put_volume')
                if call_volume and put_volume:
                    put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            squeeze_potential = 0.0
            
            # Score based on short interest percentage
            if si_percent > HIGH_SHORT_INTEREST:
                squeeze_potential += min(si_percent / HIGH_SHORT_INTEREST, 3) * 0.4
                
            # Score based on days to cover
            if days_to_cover > HIGH_DAYS_TO_COVER:
                squeeze_potential += min(days_to_cover / HIGH_DAYS_TO_COVER, 3) * 0.3
                
            # Score based on call options activity
            if call_volume is not None and put_call_ratio is not None:
                if put_call_ratio < 0.7:  # More calls than puts (bullish)
                    squeeze_potential += (1 - min(put_call_ratio, 1)) * 0.2
            
            # Score based on recent price and volume action
            # Get 10-day price and volume data
            recent_data = stock_data[stock_data['symbol'] == symbol].sort_values('Date').tail(10)
            if len(recent_data) > 5:
                # Check for increasing volume trend
                vol_change = recent_data['Volume'].pct_change().mean()
                if vol_change > 0.1:  # Volume increasing
                    squeeze_potential += min(vol_change, 1) * 0.1
                
                # Check for positive price momentum
                price_change = recent_data['Close'].pct_change().mean()
                if price_change > 0:  # Price increasing
                    squeeze_potential += min(price_change * 10, 1) * 0.1
            
            # Add metrics to result
            metrics = {
                'symbol': symbol,
                'short_interest_percent': si_percent,
                'days_to_cover': days_to_cover,
                'short_value': short_value,
                'current_price': price,
                'average_volume': volume,
                'call_volume': call_volume,
                'put_call_ratio': put_call_ratio,
                'squeeze_potential': squeeze_potential
            }
            
            squeeze_metrics.append(metrics)
        
        squeeze_df = pd.DataFrame(squeeze_metrics)
        
        # Sort by squeeze potential
        if not squeeze_df.empty:
            squeeze_df = squeeze_df.sort_values('squeeze_potential', ascending=False)
        
        logger.info(f"Calculated short squeeze metrics for {len(squeeze_df)} tickers")
        
        return squeeze_df