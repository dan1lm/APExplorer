import logging
import argparse
import time
import os
from data.collection.reddit_scraper import RedditScraper


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("short_squeeze_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(output_dir='./output', dashboard=False):
    """
    Run the pipeline to detect short squeezes.
    
    output_dir (str): Directory to save results
    dashboard (bool): Whether to launch interactive dashboard
    """
    start_time = time.time()
    logger.info("Starting detection pipeline")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect reddit data
    logger.info("Step 1: Collecting Reddit data from WSB")
    reddit_scraper = RedditScraper()        
    
    posts_df, comments_df = reddit_scraper.collect_wsb_data()
    
    if posts_df.empty:
        logger.error("Failed to collect Reddit posts. Exiting.")
        return
    
    posts_df.to_csv(os.path.join(output_dir, 'raw_posts.csv'), index=False)
    comments_df.to_csv(os.path.join(output_dir, 'raw_comments.csv'), index=False)
    
    logger.info("Step 2: Processing text to extract tickers and sentiment")
    text_processor = TextProcessor()
    ticker_mentions, ticker_sentiment, ticker_contexts = text_processor.process_posts_and_comments(posts_df, comments_df)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Short Squeeze Detector aka. APExplorer")
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch interactive dashboard')
    
    args = parser.parse_args()
    # run the pipeline
    run_pipeline(output_dir=args.output, dashboard=args.dashboard)