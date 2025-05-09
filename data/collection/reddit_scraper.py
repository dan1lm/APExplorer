import logging
import praw
from config.config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_USERNAME, REDDIT_PASSWORD, SUBREDDITS,
    POST_LIMIT, COMMENT_LIMIT, HISTORICAL_DAYS
)

from tqdm import tqdm
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

    def get_comments(self, post_id, limit=COMMENT_LIMIT):
        """
        Fetch comments for a specific post
        """
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            comments_data = []
            comment_count = 0
            
            for comment in submission.comments.list():
                if comment_count >= limit:
                    break
                    
                comment_data = {
                    'comment_id': comment.id,
                    'post_id': post_id,
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'author': str(comment.author),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                comments_data.append(comment_data)
                comment_count += 1
                
                time.sleep(0.05)
                
            return pd.DataFrame(comments_data)
        
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            return pd.DataFrame()


    def collect_wsb_data(self):
        """
        Collect posts and comments from all configured subreddits
        """
        all_posts = []
        all_comments = []
        
        for subreddit in SUBREDDITS:
            for time_filter in ['day', 'week', 'month']:
                posts_df = self.get_posts(subreddit, time_filter=time_filter)
                
                if not posts_df.empty:
                    all_posts.append(posts_df)
                    
                    for post_id in posts_df['post_id']:
                        comments_df = self.get_comments(post_id)
                        if not comments_df.empty:
                            all_comments.append(comments_df)
                            
                    time.sleep(1)
        
        # Combine all posts and comments
        posts_df = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()
        comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()
        
        logger.info(f"Total posts collected: {len(posts_df)}")
        logger.info(f"Total comments collected: {len(comments_df)}")
        
        return posts_df, comments_df



if __name__ == "__main__":
    scraper = RedditScraper()
    posts_df, comments_df = scraper.collect_wsb_data()
    
    posts_df.to_csv('wsb_posts.csv', index=False)
    comments_df.to_csv('wsb_comments.csv', index=False)