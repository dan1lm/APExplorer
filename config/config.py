import os
from dotenv import load_dotenv

load_dotenv()

#   Reddit API credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'ShortSqueezeDetector/1.0')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')

SUBREDDITS = ['wallstreetbets']

#   Data collection
POST_LIMIT = 30        # Number of posts to fetch
COMMENT_LIMIT = 30    # Max comments to fetch (per post)
HISTORICAL_DAYS = 7    # Days of historical data for analysis

#   TWELVEDATA
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

#   Text analysis settings
MIN_TICKER_MENTIONS = 5  # Minimum mentions to consider a ticker
TICKER_CONTEXT_WINDOW = 100  # Characters around ticker mention for context