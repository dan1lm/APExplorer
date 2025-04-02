import logging
import praw
from config.config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_USERNAME, REDDIT_PASSWORD, SUBREDDITS,
    POST_LIMIT, COMMENT_LIMIT, HISTORICAL_DAYS
)

import tqdm
from datetime import datetime, timedelta
import pandas as pd
import time

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
    
    def get_posts(self, subreddit_name, time_filter='week', limit=POST_LIMIT):
        """
        Fetch posts from a specific subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            time_filter (str): Time filter for posts ('day', 'week', 'month', 'year', 'all')
            limit (int): Maximum number of posts to fetch
            
        Returns:
            pandas.DataFrame: DataFrame containing post data
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            logger.info(f"Fetching {limit} posts from r/{subreddit_name} (time filter: {time_filter})")
            
            posts = subreddit.top(time_filter=time_filter, limit=limit)
            
            for post in tqdm(posts, total=limit, desc=f"Fetching posts from r/{subreddit_name}"):
                post_date = datetime.fromtimestamp(post.created_utc)
                if post_date < datetime.now() - timedelta(days=HISTORICAL_DAYS):
                    continue
                
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'author': str(post.author),
                    'url': post.url,
                    'is_self': post.is_self,
                    'subreddit': subreddit_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                posts_data.append(post_data)
                
                time.sleep(0.1)
                
            logger.info(f"Successfully fetched {len(posts_data)} posts from r/{subreddit_name}")
            return pd.DataFrame(posts_data)
        
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return pd.DataFrame()