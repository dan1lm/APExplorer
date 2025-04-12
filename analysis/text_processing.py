import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import pandas as pd
import yfinance as yf

import requests
import json
import os
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import TWELVEDATA_API_KEY

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")
    


class TextProcessor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.valid_tickers = self._load_ticker_list()
    
    def fetch_all_pages(self, base_url, max_pages=10):
        """
        Helper function to handle pagination in and rate limit
        
        Args:
            base_url (str): Base URL for the API request
            max_pages (int): Maximum number of pages to fetch
            
        Returns:
            list: Combined data from all pages
        """
        all_data = []
        
        for page in range(1, max_pages + 1):
            url = f"{base_url}&page={page}"
            
            try:
                response = requests.get(url)
                
                if response.status_code != 200:
                    logger.warning(f"API returned status code {response.status_code} for page {page}")
                    break
                    
                data = response.json()
                if 'data' not in data or not data['data']:
                    break
                    
                all_data.extend(data['data'])
                
                if page >= data.get('total_pages', 1):
                    break
                    
                # Rate limits
                time.sleep(0.25)  
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
        
        return all_data
    
    def _load_ticker_list(self):
        """
        Load a list of valid stock tickers from Twelvedata API
        
        Returns:
            set: Set of valid ticker symbols
        """
        try:
            # Try to load from cache first to avoid API calls
            cache_file = 'tickers_cache.json'
            if os.path.exists(cache_file):
                # Check if cache is recent
                file_time = os.path.getmtime(cache_file)
                if (time.time() - file_time) < 604800:  # 7 days in seconds
                    with open(cache_file, 'r') as f:
                        cached_tickers = json.load(f)
                        logger.info(f"Loaded {len(cached_tickers)} tickers from cache")
                        return set(cached_tickers)
            
            tickers = set()
            
            if not TWELVEDATA_API_KEY:
                logger.warning("TWELVEDATA_API_KEY not found in config")
                return self._get_fallback_tickers()
                
            # Get NYSE tickers
            logger.info("Fetching NYSE tickers from Twelvedata...")
            nyse_url = f"https://api.twelvedata.com/stocks?exchange=NYSE&apikey={TWELVEDATA_API_KEY}"
            nyse_stocks = self.fetch_all_pages(nyse_url, max_pages=5)
            
            # Get NASDAQ tickers
            logger.info("Fetching NASDAQ tickers from Twelvedata...")
            nasdaq_url = f"https://api.twelvedata.com/stocks?exchange=NASDAQ&apikey={TWELVEDATA_API_KEY}"
            nasdaq_stocks = self.fetch_all_pages(nasdaq_url, max_pages=5)
            
            # Extract symbols and add to set
            for stock in nyse_stocks + nasdaq_stocks:
                tickers.add(stock['symbol'])
            
            # Save to cache
            if tickers:
                with open(cache_file, 'w') as f:
                    json.dump(list(tickers), f)
            
            logger.info(f"Loaded {len(tickers)} valid ticker symbols from Twelvedata")
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading ticker list from Twelvedata: {e}")
            return self._get_fallback_tickers()
    
    def _get_fallback_tickers(self):
        """
        Fallback list of common tickers if something goes wrong with API
        
        Returns:
            set: Set of tickers
        """

        fallback_tickers = {
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", 
            "GME", "AMC", "BB", "NOK", "PLTR", "TLRY", "SNDL", "CLOV", "WISH",
            "CLNE", "WKHS", "SPCE", "CRSR", "DKNG", "RKT", "MVIS", "RIDE", 
            "SKLZ", "COIN", "SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE",
            "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "ARKX", "JPM", "BAC", 
            "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "SQ", "DIS", "NFLX",
            "ROKU", "ZM", "TDOC", "SHOP", "JNJ", "PFE", "MRNA", "BNTX", "INTC",
            "AMD", "MU", "TSM", "QCOM", "BABA", "NIO", "XPEV", "LI", "F", "GM",
            "X", "CLF", "AA", "FCX", "VALE", "MT", "GLD", "SLV", "USO", "WEED",
            "CGC", "ACB", "APHA", "TLRY", "MJ", "RIOT", "MARA", "SI", "MSTR",
            "PLBY", "UWMC", "PRPL", "PLUG", "FCEL", "BLNK", "QS", "CLOV", "HD",
            "WMT", "TGT", "COST", "KO", "PEP", "MCD", "SBUX", "DASH", "ABNB"
        }
        logger.info(f"Using fallback list of {len(fallback_tickers)} common tickers")
        return fallback_tickers
    
    def _is_valid_ticker(self, ticker):
        """
        Check if a ticker is valid and exists
        
        Args:
            ticker (str): Ticker symbol to check
            
        Returns:
            bool: True if valid, False if invalid
        """

        if ticker in self.valid_tickers:
            return True
            
        # Validate with Yahoo Finance if not in the valid tickers list
        try:
            ticker_info = yf.Ticker(ticker).info
            if 'symbol' in ticker_info:
                # Add to our valid tickers set for future reference
                self.valid_tickers.add(ticker)
                return True
            return False
        except:
            return False

if __name__ == '__main__':
    textProcessor = TextProcessor()
    print(textProcessor.valid_tickers)