import os
from dotenv import load_dotenv

load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'ShortSqueezeDetector/1.0')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')

SUBREDDITS = ['wallstreetbets']

# Data collection
POST_LIMIT = 1        # Number of posts to fetch
COMMENT_LIMIT = 1    # Max comments to fetch (per post)
HISTORICAL_DAYS = 5    # Days of historical data for analysis