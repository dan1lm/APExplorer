import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging


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
    
    def _load_ticker_list(self):
        pass