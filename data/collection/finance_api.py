import pandas as pd
import requests
import logging
import time
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
import nasdaqdatalink
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import yfinance as yf


from config.config import (
    TWELVEDATA_API_KEY,
    NASDAQ_DATA_LINK_API_KEY,
    HISTORICAL_DAYS,
    CACHE_EXPIRY_DAYS
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinanceDataCollector:
    def __init__(self):
        self.twelvedata_api_key = TWELVEDATA_API_KEY
        self.nasdaq_api_key = NASDAQ_DATA_LINK_API_KEY
        nasdaqdatalink.ApiConfig.api_key = self.nasdaq_api_key
        self.nasdaq_data_link = nasdaqdatalink
        os.makedirs('cache', exist_ok=True)
        
    def _get_cache_path(self, ticker, data_type):
        """Generate cache file path for a ticker and data type"""
        return f"cache/{ticker}_{data_type}.json"
    
    def _is_cache_valid(self, cache_path):
        """Check if cache file exists and is recent enough"""
        if not os.path.exists(cache_path):
            return False
            
        file_time = os.path.getmtime(cache_path)
        return (time.time() - file_time) < (CACHE_EXPIRY_DAYS * 86400)  # Convert to seconds
        
    def _read_from_cache(self, ticker, data_type):
        """Read data from cache if available and recent"""
        cache_path = self._get_cache_path(ticker, data_type)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for {ticker} {data_type}: {e}")
                
        return None
        
    def _write_to_cache(self, ticker, data_type, data):
        """Write data to cache file with proper JSON serialization"""
        cache_path = self._get_cache_path(ticker, data_type)

        try:
            if isinstance(data, list) and data and isinstance(data[0], dict):

                serializable_data = []
                for item in data:
                    serializable_item = {}
                    for key, value in item.items():
                        # Pandas Timestamp objects
                        if pd.api.types.is_datetime64_any_dtype(type(value)):
                            serializable_item[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                        # Numpy data types
                        elif hasattr(value, 'dtype') and hasattr(value, 'item'):
                            serializable_item[key] = value.item()  
                        else:
                            serializable_item[key] = value
                    serializable_data.append(serializable_item)

                with open(cache_path, 'w') as f:
                    json.dump(serializable_data, f)

            elif isinstance(data, dict):
                serializable_data = {}
                for key, value in data.items():
                    # Pandas Timestamp objects
                    if pd.api.types.is_datetime64_any_dtype(type(value)):
                        serializable_data[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                    # Numpy data types
                    elif hasattr(value, 'dtype') and hasattr(value, 'item'):
                        serializable_data[key] = value.item()
                    else:
                        serializable_data[key] = value

                with open(cache_path, 'w') as f:
                    json.dump(serializable_data, f)
            else:
                logger.warning(f"Unsupported data type for caching: {type(data)}")
        except Exception as e:
            logger.warning(f"Error writing cache for {ticker} {data_type}: {e}")

    def get_stock_data(self, ticker_symbol, days=HISTORICAL_DAYS):
        """
        Get historical stock data for a specific ticker using Twelvedata
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame: DataFrame with historical stock data
        """
        # Try to get from cache first
        cached_data = self._read_from_cache(ticker_symbol, "stock_data")
        if cached_data:
            try:
                return pd.DataFrame(cached_data)
            except:
                pass  # Continue to API call
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            try:
                ticker = yf.Ticker(ticker_symbol)
                hist_data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                           end=end_date.strftime('%Y-%m-%d'))
                
                if not hist_data.empty:
                    # Reset index to have date as column
                    hist_data = hist_data.reset_index()
                    # Add ticker column
                    hist_data['symbol'] = ticker_symbol
                    
                    hist_data = hist_data.rename(columns={
                        'Date': 'Date',
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    })
                    
                    # Cache the data
                    self._write_to_cache(ticker_symbol, "stock_data", hist_data.to_dict('records'))
                    
                    logger.info(f"Successfully fetched historical data for {ticker_symbol} using yfinance")
                    return hist_data
            except Exception as e:
                logger.warning(f"yfinance failed, trying Twelvedata as fallback: {e}")
            
            # Twelvedata fallback
            
            
            url = f"https://api.twelvedata.com/time_series"
            params = {
                "symbol": ticker_symbol,
                "interval": "1day",
                "outputsize": days,
                "apikey": self.twelvedata_api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get data for {ticker_symbol}: HTTP {response.status_code}")
                return None
                
            data = response.json()
            
            if "values" not in data:
                logger.warning(f"No values returned for {ticker_symbol}")
                return None
                
            hist_data = pd.DataFrame(data["values"])
            hist_data = hist_data.rename(columns={
                "datetime": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Convert data types
            for col in ["Open", "High", "Low", "Close"]:
                hist_data[col] = pd.to_numeric(hist_data[col])
            
            hist_data["Volume"] = pd.to_numeric(hist_data["Volume"], errors='coerce').fillna(0).astype(int)
            hist_data["Date"] = pd.to_datetime(hist_data["Date"])
            hist_data["symbol"] = ticker_symbol
            
            # Cache the data
            self._write_to_cache(ticker_symbol, "stock_data", hist_data.to_dict('records'))
            
            logger.info(f"Successfully fetched historical data for {ticker_symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker_symbol}: {e}")
            return None
    
    def get_short_interest_data(self, ticker_symbol):
        """
        Get short interest data for a specific ticker using NASDAQ Data Link (estimates if not available)
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            dict: Short interest data
        """
        # Try to get from cache first
        cached_data = self._read_from_cache(ticker_symbol, "short_interest")
        if cached_data:
            return cached_data
        
        # Try to get data from yfinance first if available

        try:
            ticker = yf.Ticker(ticker_symbol)
            key_stats = ticker.info
            
            # Extract relevant short interest metrics
            short_percent = key_stats.get('shortPercentOfFloat')
            
            if short_percent is not None:
                short_data = {
                    'symbol': ticker_symbol,
                    'short_interest': key_stats.get('sharesShort'),
                    'short_percent_of_float': short_percent * 100 if short_percent < 1 else short_percent,  # Convert decimal to percentage if needed
                    'days_to_cover': key_stats.get('shortRatio'),
                    'total_shares': key_stats.get('sharesOutstanding'),
                    'float_shares': key_stats.get('floatShares'),
                    'data_source': 'yfinance',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Cache the data
                self._write_to_cache(ticker_symbol, "short_interest", short_data)
                
                logger.info(f"Successfully fetched Yahoo Finance short interest data for {ticker_symbol}")
                return short_data
        except Exception as e:
            logger.warning(f"Error fetching Yahoo Finance short interest: {e}")
                
        # NASDAQ Data Link    
        if self.has_nasdaq and self.nasdaq_api_key:
            try:
                # Short interest data from NASDAQ
                short_data = self.nasdaq_data_link.get_table('FINRA/SHORTS', 
                                                      date={'gte': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')},
                                                      ticker=ticker_symbol)
                
                if not short_data.empty:
                    # Get most recent data point
                    latest = short_data.iloc[0]
                    
                    # Calculate days to cover
                    avg_volume = latest.get('avg_daily_volume', 0)
                    days_to_cover = latest['short_interest'] / avg_volume if avg_volume > 0 else None
                    
                    si_data = {
                        'symbol': ticker_symbol,
                        'short_interest': int(latest['short_interest']),
                        'short_percent_of_float': float(latest['short_percent_of_float']),
                        'days_to_cover': float(days_to_cover) if days_to_cover else None,
                        'total_shares': int(latest.get('outstanding_shares', 0)),
                        'float_shares': int(latest.get('float_shares', 0)),
                        'data_source': 'nasdaq',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Cache the data
                    self._write_to_cache(ticker_symbol, "short_interest", si_data)
                    
                    logger.info(f"Successfully fetched NASDAQ short interest data for {ticker_symbol}")
                    return si_data
                    
            except Exception as e:
                logger.warning(f"Error fetching NASDAQ short interest for {ticker_symbol}: {e}")
        
        
        # Twelvedata's available data fallback
        try:
            # Get fundamentals from Twelvedata
            fundamentals_url = f"https://api.twelvedata.com/fundamentals?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
            response = requests.get(fundamentals_url)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get fundamentals for {ticker_symbol}: HTTP {response.status_code}")
                return self._create_empty_short_data(ticker_symbol)
            
            data = response.json()
            
            if not data or "fundamentals" not in data:
                logger.warning(f"No fundamental data available for {ticker_symbol}")
                return self._create_empty_short_data(ticker_symbol)
            
            fundamentals = data.get("fundamentals", {})
            highlights = fundamentals.get("highlights", {})
            shares_stats = fundamentals.get("shares_statistics", {})
            
            # Extract available data
            total_shares = shares_stats.get("shares_outstanding", None)
            float_shares = shares_stats.get("shares_float", None)
            
            # Get trading volume
            quote_url = f"https://api.twelvedata.com/quote?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
            quote_response = requests.get(quote_url)
            
            volume = None
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                if "volume" in quote_data:
                    volume = int(quote_data["volume"])
            
            # Estimate short interet
            est_short_percent = 5.0  # Default estimate
            short_interest = None
            
            if float_shares:
                short_interest = int(float(float_shares) * (est_short_percent / 100))
                
            # Estimate days to cover
            days_to_cover = None
            if short_interest and volume and volume > 0:
                days_to_cover = short_interest / volume
            
            short_data = {
                'symbol': ticker_symbol,
                'short_interest': short_interest,
                'short_percent_of_float': est_short_percent,
                'days_to_cover': days_to_cover,
                'total_shares': total_shares,
                'float_shares': float_shares,
                'data_source': 'estimate',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Cache the data
            self._write_to_cache(ticker_symbol, "short_interest", short_data)
            
            logger.info(f"Created estimated short interest data for {ticker_symbol}")
            return short_data
            
        except Exception as e:
            logger.error(f"Error estimating short interest data for {ticker_symbol}: {e}")
            return self._create_empty_short_data(ticker_symbol)
    
    def _create_empty_short_data(self, ticker_symbol):
        """Helper to create empty short interest data structure"""
        return {
            'symbol': ticker_symbol,
            'short_interest': None,
            'short_percent_of_float': None,
            'days_to_cover': None,
            'total_shares': None,
            'float_shares': None,
            'data_source': 'none',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_options_data(self, ticker_symbol):
        """
        Get options data for a specific ticker with Yahoo Finance
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            
        Returns:
            dict: Options activity data
        """
        # Try to get from cache first
        cached_data = self._read_from_cache(ticker_symbol, "options")
        if cached_data:
            return cached_data
            
        try:
            ticker = yf.Ticker(ticker_symbol)
        
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options data available for {ticker_symbol}")
                return self._create_estimated_options_data(ticker_symbol)
                
            nearest_exp = expirations[0]
            
            # Get calls and puts for the nearest expiration
            options = ticker.option_chain(nearest_exp)
            calls = options.calls
            puts = options.puts
            
            # Calculate put/call ratio
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            # Calculate total call open interest
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            
            options_data = {
                'symbol': ticker_symbol,
                'expiration_date': nearest_exp,
                'call_volume': int(call_volume),
                'put_volume': int(put_volume),
                'put_call_ratio': float(pc_ratio),
                'call_open_interest': int(call_oi),
                'put_open_interest': int(put_oi),
                'data_source': 'yfinance',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Cache the data
            self._write_to_cache(ticker_symbol, "options", options_data)
            
            logger.info(f"Successfully fetched Yahoo Finance options data for {ticker_symbol}")
            return options_data
            
        except Exception as e:
            logger.warning(f"Error fetching options data from Yahoo Finance: {e}")
            # Fall back to estimates
        
        # Fall back to estimates
        return self._create_estimated_options_data(ticker_symbol)
    
    def _create_estimated_options_data(self, ticker_symbol):
        """Create estimated options data based on stock metrics"""
        try:
            # Get stock stats from Twelvedata or use yfinance as fallback
            stock_data = None
            
            # Try Twelvedata first
            stats_url = f"https://api.twelvedata.com/statistics?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
            response = requests.get(stats_url)
            
            avg_volume = 0
            if response.status_code == 200:
                data = response.json()
                if "statistics" in data:
                    avg_volume = int(data.get("statistics", {}).get("average_volume", 0))
            
            if avg_volume == 0:
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    avg_volume = ticker.info.get('averageVolume', 0)
                except:
                    pass
            
            # Set default values
            next_month = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Create estimated data
            options_data = {
                'symbol': ticker_symbol,
                'expiration_date': next_month,
                'call_volume': int(avg_volume * 0.15),  
                'put_volume': int(avg_volume * 0.10),  
                'put_call_ratio': 0.67,  
                'call_open_interest': int(avg_volume * 0.5),  
                'put_open_interest': int(avg_volume * 0.3),  
                'data_source': 'estimate',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Calculate put/call ratio
            if options_data['call_volume'] > 0:
                options_data['put_call_ratio'] = options_data['put_volume'] / options_data['call_volume']
            
            # Cache the data
            self._write_to_cache(ticker_symbol, "options", options_data)
            
            logger.info(f"Created estimated options data for {ticker_symbol}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error creating estimated options data for {ticker_symbol}: {e}")
            
            # Return basic estimates if all else fails
            return {
                'symbol': ticker_symbol,
                'expiration_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'call_volume': 1000,
                'put_volume': 700,
                'put_call_ratio': 0.7,
                'call_open_interest': 5000,
                'put_open_interest': 3500,
                'data_source': 'fallback_estimate',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def is_valid_ticker(self, ticker_symbol):
        """
        Check if a ticker symbol is valid
        
        Args:
            ticker_symbol (str): Ticker symbol to check
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Try using yfinance first if available (most reliable)
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except:
            pass  # Fall back to Twelvedata
        
        # Fall back to Twelvedata
        try:
            url = f"https://api.twelvedata.com/quote?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return False
                
            data = response.json()
            
            # If the response contains an error field, the ticker is invalid
            if "code" in data and data["code"] == 400:
                return False
                
            # If symbol is in the response, the ticker is valid
            return "symbol" in data
            
        except Exception:
            return False
    
    def collect_all_data(self, ticker_list):
        """
        Collect all financial data for a list of tickers
        
        Args:
            ticker_list (list): List of ticker symbols
            
        Returns:
            tuple: (stock_data, short_interest_data, options_data)
        """
        stock_data = []
        short_interest_data = []
        options_data = []
        
        # Filter for valid tickers first
        valid_tickers = []
        logger.info("Validating tickers...")
        for ticker in tqdm(ticker_list, desc="Validating tickers"):
            if self.is_valid_ticker(ticker):
                valid_tickers.append(ticker)

            time.sleep(0.25)
        
        logger.info(f"Found {len(valid_tickers)} valid tickers out of {len(ticker_list)}")
        
        for ticker in tqdm(valid_tickers, desc="Collecting financial data"):
            hist_data = self.get_stock_data(ticker)
            if hist_data is not None:
                stock_data.append(hist_data)
            
            # Get short interest data
            si_data = self.get_short_interest_data(ticker)
            if si_data:
                short_interest_data.append(si_data)
            
            # Get options data
            opt_data = self.get_options_data(ticker)
            if opt_data:
                options_data.append(opt_data)
            
            time.sleep(0.5)
        
        stock_df = pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame()
        short_interest_df = pd.DataFrame(short_interest_data) if short_interest_data else pd.DataFrame()
        options_df = pd.DataFrame(options_data) if options_data else pd.DataFrame()
        
        logger.info(f"Collected data for {len(valid_tickers)} tickers")
        
        return stock_df, short_interest_df, options_df


if __name__ == "__main__":
    test_tickers = ['AAPL', 'MSFT', 'TSLA', 'GME', 'AMC']
    collector = FinanceDataCollector()
    stock_df, short_interest_df, options_df = collector.collect_all_data(test_tickers)
    
    if not stock_df.empty:
        stock_df.to_csv('stock_data.csv', index=False)
    if not short_interest_df.empty:
        short_interest_df.to_csv('short_interest_data.csv', index=False)
    if not options_df.empty:
        options_df.to_csv('options_data.csv', index=False)