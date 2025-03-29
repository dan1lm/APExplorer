import logging
import praw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedditScraper:
    def __init__(self):
        """Reddit API connection initialization"""
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
                username=REDDIT_USERNAME,
                password=REDDIT_PASSWORD
            )
            logger.info("Successfully connected to Reddit API")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise