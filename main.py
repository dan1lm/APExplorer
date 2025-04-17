import logging
import argparse
import time
import os
from data.collection.reddit_scraper import RedditScraper
from analysis.text_processing import TextProcessor
import pandas as pd
from data.collection.finance_api import FinanceDataCollector
from analysis.financial_metrics import FinancialMetricsAnalyzer

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
    
    
    
    ########## TEXT PROCESSOR   ##########
    logger.info("Step 2: Processing text to extract tickers and sentiment")
    text_processor = TextProcessor()
    ticker_mentions, ticker_sentiment, ticker_contexts = text_processor.process_posts_and_comments(posts_df, comments_df)
    
    mentions_df = pd.DataFrame(list(ticker_mentions.items()), columns=['symbol', 'mentions'])
    sentiment_df = pd.DataFrame([(ticker, avg) for ticker, avg in ticker_sentiment.items()], columns=['symbol', 'sentiment'])
    ticker_data = pd.merge(mentions_df, sentiment_df, on='symbol', how='outer')
    ticker_data.to_csv(os.path.join(output_dir, 'ticker_data.csv'), index=False)
    
    contexts_data = []
    for ticker, contexts_list in ticker_contexts.items():
        for context in contexts_list:
            contexts_data.append({'symbol': ticker, 'context': context})
    
    pd.DataFrame(contexts_data).to_csv(os.path.join(output_dir, 'ticker_contexts.csv'), index=False)


    # Collecting financial data for the tickers
    logger.info(f"Step 3: Collecting financial data for {len(ticker_mentions)} tickers")
    finance_collector = FinanceDataCollector()
    stock_df, short_interest_df, options_df = finance_collector.collect_all_data(list(ticker_mentions.keys()))
    
    # Save financial data
    stock_df.to_csv(os.path.join(output_dir, 'stock_data.csv'), index=False)
    short_interest_df.to_csv(os.path.join(output_dir, 'short_interest_data.csv'), index=False)
    options_df.to_csv(os.path.join(output_dir, 'options_data.csv'), index=False)
    
    
    # Financial metrics calculations
    logger.info("Step 4: Calculating financial metrics")
    metrics_analyzer = FinancialMetricsAnalyzer()
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Short Squeeze Detector aka. APExplorer")
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch interactive dashboard')
    
    args = parser.parse_args()
    # run the pipeline
    run_pipeline(output_dir=args.output, dashboard=args.dashboard)