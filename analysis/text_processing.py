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
import re
from tqdm import tqdm

from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import (
    MIN_TICKER_MENTIONS, 
    TICKER_CONTEXT_WINDOW,
    TWELVEDATA_API_KEY
)

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
    
    
    def extract_tickers(self, text):
        """
        Extract stock ticker symbols from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of extracted ticker symbols
        """
        if not text or not isinstance(text, str):
            return []
            
        # Ticker patterns
        # $TICKER
        pattern1 = r'\$([A-Z]{1,5})\b'
        pattern2 = r'\b([A-Z]{1,5})\b'
        
        p1_tickers = re.findall(pattern1, text)
        p2_tickers = re.findall(pattern2, text)
        
        # Combine and filter
        potential_tickers = set(p1_tickers + p2_tickers)
        
        # preference: $ prefixed
        valid_tickers = []
        for ticker in potential_tickers:
            if ticker in p1_tickers or self._is_valid_ticker(ticker):
                # For non-$ prefixed tickers: avoid common abbreviations
                if ticker not in ['A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'IPO', 'ATH', 'DD', 'FD', 'YOLO']:
                    valid_tickers.append(ticker)
        
        return valid_tickers
    
    
    def get_ticker_context(self, text, ticker, window_size=TICKER_CONTEXT_WINDOW):
        """
        Extract context window around ticker mentions
        
        Args:
            text (str): Source text
            ticker (str): Ticker symbol
            window_size (int): Character window size (each side)
            
        Returns:
            list: List of context strings around ticker mentions
        """
        if not text or not isinstance(text, str):
            return []
            
        contexts = []
        
        # Formats: $TICKER and TICKER
        patterns = [f'\\${ticker}\\b', f'\\b{ticker}\\b']
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                context = text[start:end]
                contexts.append(context)
                
        return contexts
    
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not text or not isinstance(text, str):
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
        # WSB-specific sentiment adjustments
        wsb_text = text
    
        wsb_terms = {
            'moon': 2.0,
            'rocket': 2.0,
            'tendies': 1.5,
            'diamond hands': 1.5,
            'paper hands': -1.0,
            'yolo': 1.0,
            'short squeeze': 1.5,
            'gamma squeeze': 1.5,
            'bear': -0.5,
            'bull': 0.5,
            'calls': 0.5,
            'puts': -0.5,
            'ape': 1.0,
            'hodl': 1.0,
            'bagholder': -1.0,
        }
        
        # Calculate basic sentiment
        sentiment = self.sia.polarity_scores(wsb_text)
        
        # Adjust for specific terms
        compound_adjustment = 0
        for term, score in wsb_terms.items():
            if term in wsb_text.lower():
                compound_adjustment += score * 0.1
        
        # Adjustment to compound score
        sentiment['compound'] = max(-1.0, min(1.0, sentiment['compound'] + compound_adjustment))
        
        return sentiment
    
    
    def process_posts_and_comments(self, posts_df, comments_df):
        """
        Process posts and comments to extract tickers and sentiment
        
        Args:
            posts_df (pandas.DataFrame): DataFrame of Reddit posts
            comments_df (pandas.DataFrame): DataFrame of Reddit comments
            
        Returns:
            tuple: (ticker_mentions, ticker_sentiment, ticker_contexts)
        """

        posts_text = []
        if 'title' in posts_df.columns and 'text' in posts_df.columns:
            posts_text = posts_df['title'] + ' ' + posts_df['text'].fillna('')
        
        comments_text = []
        if 'text' in comments_df.columns:
            comments_text = comments_df['text'].fillna('')
        
        all_texts = list(posts_text) + list(comments_text)
        
        # Extract tickers
        ticker_mentions = Counter()
        ticker_sentiment = {}
        ticker_contexts = {}
        
        logger.info("Processing text to extract tickers and sentiment")
        
        for text in tqdm(all_texts, desc="Processing text"):
            tickers = self.extract_tickers(text)
            
            for ticker in tickers:
                ticker_mentions[ticker] += 1
                
                # Get context around ticker
                contexts = self.get_ticker_context(text, ticker)
                
                if ticker not in ticker_contexts:
                    ticker_contexts[ticker] = []
                ticker_contexts[ticker].extend(contexts)
                
                # Analyze sentiment in the context
                for context in contexts:
                    sentiment = self.analyze_sentiment(context)
                    
                    if ticker not in ticker_sentiment:
                        ticker_sentiment[ticker] = []
                    ticker_sentiment[ticker].append(sentiment['compound'])
        
        filtered_tickers = {ticker: count for ticker, count in ticker_mentions.items() 
                           if count >= MIN_TICKER_MENTIONS}
        
        # Calculate average sentiment for each ticker
        avg_sentiment = {}
        for ticker, sentiments in ticker_sentiment.items():
            if ticker in filtered_tickers:
                avg_sentiment[ticker] = sum(sentiments) / len(sentiments)
        
        # Keep only contexts for filtered tickers
        filtered_contexts = {ticker: contexts for ticker, contexts in ticker_contexts.items() 
                            if ticker in filtered_tickers}
        
        logger.info(f"Found {len(filtered_tickers)} tickers with {MIN_TICKER_MENTIONS}+ mentions")
        
        return filtered_tickers, avg_sentiment, filtered_contexts

if __name__ == "__main__":
    
    # Testing the text processor with sample data
    test_text = """
    I'm bullish on $GME and think it's going to moooooon 🚀🚀🚀
    Also looking at AAPL and TSLA, but those aren't as interesting.
    """
    
    processor = TextProcessor()
    
    # Testing ticker extraction
    tickers = processor.extract_tickers(test_text)
    print(f"Extracted tickers: {tickers}")
    
    # Testing sentiment analysis
    for ticker in tickers:
        contexts = processor.get_ticker_context(test_text, ticker)
        sentiments = [processor.analyze_sentiment(ctx) for ctx in contexts]
        print(f"{ticker} contexts: {contexts}")
        print(f"{ticker} sentiment scores: {[s['compound'] for s in sentiments]}")
        
    # Print the number of valid tickers loaded
    print(f"Total valid tickers loaded: {len(processor.valid_tickers)}")
    print(f"Sample tickers: {list(processor.valid_tickers)[:10]}")