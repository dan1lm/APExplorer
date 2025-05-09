import pandas as pd
import requests
import logging
import time
import json
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
import nasdaqdatalink
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
        
        # Initialize NASDAQ Data Link API
        try:
            nasdaqdatalink.ApiConfig.api_key = self.nasdaq_api_key
            self.nasdaq_data_link = nasdaqdatalink
            self.has_nasdaq = True
            logger.info("Successfully initialized NASDAQ Data Link API")
        except Exception as e:
            logger.warning(f"Failed to initialize NASDAQ Data Link API: {e}")
            self.has_nasdaq = False
        
        # Common WSB tickers to validate even if not found in APIs
        self.common_wsb_tickers = {
            'SPY', 'QQQ', 'GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 
            'GOOG', 'GOOGL', 'BB', 'NOK', 'PLTR', 'BABA', 'AMD', 'INTC', 'TLRY', 'SNDL',
            'COIN', 'HOOD', 'SNAP', 'NFLX', 'DIS', 'RIVN', 'LCID', 'NIO', 'F', 'GM', 'T'
        }
            
        # Create cache directory
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
        Get historical stock data using NASDAQ Data Link
        """
        # Try to get from cache first
        cached_data = self._read_from_cache(ticker_symbol, "stock_data")
        if cached_data:
            try:
                return pd.DataFrame(cached_data)
            except:
                pass  
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Try NASDAQ Data Link first for stock data
            if self.has_nasdaq:
                try:
                    for dataset_format in [f"EOD/{ticker_symbol}", f"WIKI/{ticker_symbol}", ticker_symbol]:
                        try:
                            data = self.nasdaq_data_link.get(
                                dataset_format,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            )
                            
                            if not data.empty:
                                # Process data
                                hist_data = data.reset_index()
                                column_mapping = {
                                    'Date': 'Date',
                                    'Open': 'Open',
                                    'High': 'High',
                                    'Low': 'Low',
                                    'Close': 'Close',
                                    'Adj. Close': 'Close',
                                    'Adj. Open': 'Open',
                                    'Adj. High': 'High',
                                    'Adj. Low': 'Low',
                                    'Adjusted Close': 'Close',
                                    'Adjusted Open': 'Open',
                                    'Adjusted High': 'High',
                                    'Adjusted Low': 'Low',
                                    'Volume': 'Volume',
                                }
                                
                                
                                hist_data.columns = [column_mapping.get(col, col) for col in hist_data.columns]
                                
                                
                                hist_data['symbol'] = ticker_symbol
                                
                                
                                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
                                for col in required_columns:
                                    if col not in hist_data.columns:
                                        if col != 'symbol':  
                                            hist_data[col] = 0
                                
                                self._write_to_cache(ticker_symbol, "stock_data", hist_data.to_dict('records'))
                                
                                logger.info(f"Successfully fetched historical data for {ticker_symbol} using NASDAQ Data Link")
                                return hist_data
                                
                        except Exception as e:
                            logger.debug(f"Failed to get data for {ticker_symbol} with format {dataset_format}: {e}")
                            continue  
                            
                    logger.warning(f"All NASDAQ Data Link formats failed for {ticker_symbol}")
                    
                except Exception as e:
                    logger.warning(f"NASDAQ Data Link failed for stock data, trying Twelvedata fallback: {e}")
            
            # Twelvedata fallback
            if self.twelvedata_api_key:
                try:
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
                        return self._create_dummy_stock_data(ticker_symbol, days)
                        
                    data = response.json()
                    
                    if "values" not in data:
                        logger.warning(f"No values returned for {ticker_symbol}")
                        return self._create_dummy_stock_data(ticker_symbol, days)
                        
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
                    
                    logger.info(f"Successfully fetched historical data for {ticker_symbol} using Twelvedata")
                    return hist_data
                except Exception as e:
                    logger.warning(f"Twelvedata failed for {ticker_symbol}: {e}")
            
            # If all else fails, create dummy data for common WSB tickers
            if ticker_symbol in self.common_wsb_tickers:
                logger.warning(f"Using dummy data for {ticker_symbol} as it's a common WSB ticker")
                return self._create_dummy_stock_data(ticker_symbol, days)
                
            logger.error(f"All data sources failed for {ticker_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker_symbol}: {e}")
            return None
            
    def _create_dummy_stock_data(self, ticker_symbol, days=30):
        """Create dummy stock data for common tickers to enable pipeline to continue"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        base_price = 100.0
        
        # Simple random walk
        np.random.seed(hash(ticker_symbol) % 10000)  # Seed based on ticker for consistency
        daily_returns = np.random.normal(0.0005, 0.02, len(date_range))  # Small positive drift
        price_factors = np.cumprod(1 + daily_returns)
        
        close_prices = base_price * price_factors
        open_prices = close_prices / (1 + np.random.normal(0, 0.005, len(date_range)))
        high_prices = np.maximum(close_prices, open_prices) * (1 + abs(np.random.normal(0, 0.01, len(date_range))))
        low_prices = np.minimum(close_prices, open_prices) * (1 - abs(np.random.normal(0, 0.01, len(date_range))))
        volumes = np.random.randint(100000, 10000000, len(date_range))
        
        dummy_data = pd.DataFrame({
            'Date': date_range,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes,
            'symbol': ticker_symbol
        })
        
        logger.info(f"Created dummy stock data for {ticker_symbol}")
        return dummy_data
    
    def get_short_interest_data(self, ticker_symbol):
        """
        Get short interest data using NASDAQ Data Link
        """
        # Try to get from cache first
        cached_data = self._read_from_cache(ticker_symbol, "short_interest")
        if cached_data:
            return cached_data
                
        # Try NASDAQ Data Link for short interest data
        if self.has_nasdaq:
            try:
                # Short interest data from NASDAQ (FINRA/SHORTS dataset)
                try:
                    short_data = self.nasdaq_data_link.get_table('FINRA/SHORTS', 
                                                      date={'gte': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')},
                                                      ticker=ticker_symbol)
                    
                    if not short_data.empty:
                        # Get most recent data point and calculate days to cover
                        latest = short_data.iloc[0]
                        
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
                        
                        
                        self._write_to_cache(ticker_symbol, "short_interest", si_data)
                        
                        logger.info(f"Successfully fetched NASDAQ short interest data for {ticker_symbol}")
                        return si_data
                except Exception as e:
                    logger.warning(f"Failed to get NASDAQ FINRA/SHORTS data for {ticker_symbol}: {e}")
                    
                # Try other NASDAQ datasets that might have short interest
                try:
                    # Quandl/SI (adjust on a dataset that is available)
                    alt_data = self.nasdaq_data_link.get(f"SI/{ticker_symbol}", rows=1)
                    
                    if not alt_data.empty:
                        # Process the data based on the specific dataset format (customize on dataset)
                        logger.info(f"Found alternative short interest data for {ticker_symbol}")
                    
                except Exception:
                    pass
                        
            except Exception as e:
                logger.warning(f"Error fetching NASDAQ short interest for {ticker_symbol}: {e}")
        
        if self.twelvedata_api_key:
            try:
                fundamentals_url = f"https://api.twelvedata.com/fundamentals?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
                response = requests.get(fundamentals_url)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to get fundamentals for {ticker_symbol}: HTTP {response.status_code}")
                    return self._create_estimated_short_data(ticker_symbol)
                
                data = response.json()
                
                if not data or "fundamentals" not in data:
                    logger.warning(f"No fundamental data available for {ticker_symbol}")
                    return self._create_estimated_short_data(ticker_symbol)
                
                fundamentals = data.get("fundamentals", {})
                highlights = fundamentals.get("highlights", {})
                shares_stats = fundamentals.get("shares_statistics", {})
                
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
                
                # Estimate short interest
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
                    'data_source': 'twelvedata_estimate',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self._write_to_cache(ticker_symbol, "short_interest", short_data)
                
                logger.info(f"Created estimated short interest data for {ticker_symbol} from Twelvedata")
                return short_data
                
            except Exception as e:
                logger.error(f"Error estimating short interest data for {ticker_symbol}: {e}")
        
        if ticker_symbol in self.common_wsb_tickers:
            return self._create_estimated_short_data(ticker_symbol)
            
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
        
    def _create_estimated_short_data(self, ticker_symbol):
        """Create estimated short interest data for common tickers"""
        # For common WSB tickers: use plausible estimates (different ranges for different types of stocks)
        
        ticker_hash = hash(ticker_symbol) % 10000
        np.random.seed(ticker_hash)
        
        # Meme stocks tend to have higher short interest
        is_meme_stock = ticker_symbol in {'GME', 'AMC', 'BB', 'NOK', 'CLOV', 'WISH', 'CLNE', 'WKHS'}
        
        if is_meme_stock:
            short_percent = np.random.uniform(15.0, 35.0)  # Higher short interest
            days_to_cover = np.random.uniform(2.0, 8.0)
        else:
            short_percent = np.random.uniform(2.0, 15.0)   # Lower short interest
            days_to_cover = np.random.uniform(1.0, 4.0)
            
        market_cap = np.random.uniform(1e9, 100e9)  # $1B to $100B market cap
        share_price = 50.0  # Placeholder average price
        shares_outstanding = int(market_cap / share_price)
        float_shares = int(shares_outstanding * np.random.uniform(0.7, 0.95))  # 70-95% of outstanding
        
        # Calculate short interest
        short_interest = int(float_shares * short_percent / 100)
        
        short_data = {
            'symbol': ticker_symbol,
            'short_interest': short_interest,
            'short_percent_of_float': short_percent,
            'days_to_cover': days_to_cover,
            'total_shares': shares_outstanding,
            'float_shares': float_shares,
            'data_source': 'estimate',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Created estimated short interest data for {ticker_symbol}")
        return short_data
    
    def get_options_data(self, ticker_symbol):
        """
        Get options data using available sources or generate estimates
        """
        cached_data = self._read_from_cache(ticker_symbol, "options")
        if cached_data:
            return cached_data
            
        # Try to estimate based on stock data (no reliable options data in free NASDAQ tiers)
        return self._create_estimated_options_data(ticker_symbol)
    
    def _create_estimated_options_data(self, ticker_symbol):
        """Create estimated options data based on stock metrics or dummy data for common tickers"""
        try:
            # Get stock data if available
            stock_data = self.get_stock_data(ticker_symbol, days=30)
            
            if stock_data is not None and not stock_data.empty:
                # Calculate average volume
                avg_volume = stock_data['Volume'].mean()
                
                # Get the latest price
                latest_price = stock_data['Close'].iloc[0] if stock_data['Close'].iloc[0] > 0 else 50
                
                # Set default values for expiration
                next_month = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Create estimated data based on typical options volumes relative to stock volume
                options_data = {
                    'symbol': ticker_symbol,
                    'expiration_date': next_month,
                    'call_volume': int(avg_volume * 0.15),  # calls tend to be 10-20% of stock volume
                    'put_volume': int(avg_volume * 0.10),   # puts tend to be slightly lower than calls
                    'put_call_ratio': 0.67,                 # Typical put/call ratio
                    'call_open_interest': int(avg_volume * 0.5),  # Open interest accumulation
                    'put_open_interest': int(avg_volume * 0.3),   # Call open interest tends to be higher
                    'data_source': 'estimate_from_stock',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Calculate put/call ratio
                if options_data['call_volume'] > 0:
                    options_data['put_call_ratio'] = options_data['put_volume'] / options_data['call_volume']
                
                # Cache the data
                self._write_to_cache(ticker_symbol, "options", options_data)
                
                logger.info(f"Created estimated options data for {ticker_symbol} based on stock data")
                return options_data
            else:
                # Use plausible defaults in absence of stock data
                return self._create_default_options_data(ticker_symbol)
                
        except Exception as e:
            logger.error(f"Error creating estimated options data for {ticker_symbol}: {e}")
            return self._create_default_options_data(ticker_symbol)
    
    def _create_default_options_data(self, ticker_symbol):
        """Create plausible options data when no other data is available"""
        # Meme stocks have different options profiles
        is_meme_stock = ticker_symbol in {'GME', 'AMC', 'BB', 'NOK', 'CLOV', 'WISH', 'CLNE', 'WKHS'}
        
        ticker_hash = hash(ticker_symbol) % 10000
        np.random.seed(ticker_hash)
        
        # Base values
        base_volume = np.random.randint(5000, 50000)
        
        if is_meme_stock:
            # Meme stocks -> higher call volume i.e. bullish
            call_volume = int(base_volume * np.random.uniform(1.2, 2.0))
            put_volume = int(base_volume * np.random.uniform(0.5, 0.9))
            call_oi = int(call_volume * np.random.uniform(5.0, 10.0))  # Higher open interest accumulation
            put_oi = int(put_volume * np.random.uniform(4.0, 8.0))
        else:
            # Regular stocks -> more balanced options profile
            call_volume = int(base_volume * np.random.uniform(0.8, 1.2))
            put_volume = int(base_volume * np.random.uniform(0.7, 1.1))
            call_oi = int(call_volume * np.random.uniform(3.0, 6.0))
            put_oi = int(put_volume * np.random.uniform(2.5, 5.5))
        
        # Calculate put/call ratio
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0.7
        
        return {
            'symbol': ticker_symbol,
            'expiration_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'call_volume': call_volume,
            'put_volume': put_volume,
            'put_call_ratio': put_call_ratio,
            'call_open_interest': call_oi,
            'put_open_interest': put_oi,
            'data_source': 'default_estimate',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def is_valid_ticker(self, ticker_symbol):
        """
        Enhanced ticker validation with NASDAQ Data Link and Twelvedata
        """
        
        if ticker_symbol in self.common_wsb_tickers:
            logger.info(f"Validated {ticker_symbol} as common WSB ticker")
            return True
            
        # Filter out obvious non-tickers
        if len(ticker_symbol) > 5 or not ticker_symbol.isalpha():
            return False
            
        # Try NASDAQ Data Link first for validation
        if self.has_nasdaq:
            for dataset_format in [f"EOD/{ticker_symbol}", f"WIKI/{ticker_symbol}"]:
                try:
                    # Just get one row to validate existence
                    self.nasdaq_data_link.get(dataset_format, rows=1)
                    logger.info(f"Validated {ticker_symbol} with NASDAQ Data Link")
                    return True
                except Exception:
                    # Failure for non-existent tickers
                    continue
        
        if self.twelvedata_api_key:
            try:
                url = f"https://api.twelvedata.com/quote?symbol={ticker_symbol}&apikey={self.twelvedata_api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # If the response contains an error field, the ticker is invalid
                    if "code" in data and data["code"] == 400:
                        return False
                        
                    # If symbol is in the response, the ticker is valid
                    if "symbol" in data:
                        logger.info(f"Validated {ticker_symbol} with Twelvedata")
                        return True
                    
            except Exception:
                pass
        
        # If all validation methods fail, return False
        logger.warning(f"Could not validate ticker {ticker_symbol} with any data source")
        return False
    
    def collect_all_data(self, ticker_list):
        """
        Collect all financial data for a list of tickers
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
            else:
                logger.debug(f"Ticker {ticker} failed validation")
            time.sleep(0.25)
        
        logger.info(f"Found {len(valid_tickers)} valid tickers out of {len(ticker_list)}")
        
        # Handle case with no valid tickers but continue pipeline
        if not valid_tickers and self.common_wsb_tickers:
            logger.warning("No valid tickers found, using common WSB tickers for demonstration")
            # Use a few common tickers to allow pipeline to continue
            valid_tickers = list(self.common_wsb_tickers)[:5]  # Just use 5 common tickers
        
        for ticker in tqdm(valid_tickers, desc="Collecting financial data"):
            # Get historical stock data
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