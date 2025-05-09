import logging
import argparse
import time
import os
from data.collection.reddit_scraper import RedditScraper
from analysis.text_processing import TextProcessor
import pandas as pd
from data.collection.finance_api import FinanceDataCollector
from analysis.financial_metrics import FinancialMetricsAnalyzer
from analysis.scoring import SqueezeScorer
from visualization.dashboard import SqueezeVisualizer

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
    Run the complete pipeline to detect short squeeze candidates
    """
    start_time = time.time()
    logger.info("Starting APExplorer pipeline")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Collect Reddit data
    logger.info("Step 1: Collecting Reddit data from WSB")
    reddit_scraper = RedditScraper()
    posts_df, comments_df = reddit_scraper.collect_wsb_data()
    
    if posts_df.empty:
        logger.error("Failed to collect Reddit posts. Exiting.")
        return
    
    posts_df.to_csv(os.path.join(output_dir, 'raw_posts.csv'), index=False)
    comments_df.to_csv(os.path.join(output_dir, 'raw_comments.csv'), index=False)
    
    # Step 2: Process text to extract tickers and sentiment
    logger.info("Step 2: Processing text to extract tickers and sentiment")
    text_processor = TextProcessor()
    ticker_mentions, ticker_sentiment, ticker_contexts = text_processor.process_posts_and_comments(posts_df, comments_df)
    
    # Save ticker mentions and sentiment
    mentions_df = pd.DataFrame(list(ticker_mentions.items()), columns=['symbol', 'mentions'])
    sentiment_df = pd.DataFrame([(ticker, avg) for ticker, avg in ticker_sentiment.items()], columns=['symbol', 'sentiment'])
    ticker_data = pd.merge(mentions_df, sentiment_df, on='symbol', how='outer')
    ticker_data.to_csv(os.path.join(output_dir, 'ticker_data.csv'), index=False)
    
    # Save contexts for reference
    contexts_data = []
    for ticker, contexts_list in ticker_contexts.items():
        for context in contexts_list:
            contexts_data.append({'symbol': ticker, 'context': context})
    
    pd.DataFrame(contexts_data).to_csv(os.path.join(output_dir, 'ticker_contexts.csv'), index=False)
    
    # Step 3: Collect financial data for mentioned tickers
    logger.info(f"Step 3: Collecting financial data for {len(ticker_mentions)} tickers")
    finance_collector = FinanceDataCollector()
    stock_df, short_interest_df, options_df = finance_collector.collect_all_data(list(ticker_mentions.keys()))
    
    # Save financial data
    if not stock_df.empty:
        stock_df.to_csv(os.path.join(output_dir, 'stock_data.csv'), index=False)
    if not short_interest_df.empty:
        short_interest_df.to_csv(os.path.join(output_dir, 'short_interest_data.csv'), index=False)
    if not options_df.empty:
        options_df.to_csv(os.path.join(output_dir, 'options_data.csv'), index=False)
    
    # Step 4: Calculate financial metrics
    logger.info("Step 4: Calculating financial metrics")
    metrics_analyzer = FinancialMetricsAnalyzer()
    
    # Check if we have financial data to analyze
    if stock_df.empty:
        logger.warning("No stock data available. Using common fallback stocks.")
        # Fallback to common stocks to allow pipeline to continue
        fallback_tickers = ['SPY', 'GME', 'AMC', 'TSLA', 'AAPL']
        stock_df, short_interest_df, options_df = finance_collector.collect_all_data(fallback_tickers)
        
        # Save fallback financial data
        if not stock_df.empty:
            stock_df.to_csv(os.path.join(output_dir, 'fallback_stock_data.csv'), index=False)
        if not short_interest_df.empty:
            short_interest_df.to_csv(os.path.join(output_dir, 'fallback_short_interest_data.csv'), index=False)
        if not options_df.empty:
            options_df.to_csv(os.path.join(output_dir, 'fallback_options_data.csv'), index=False)
    
    # Calculate volatility metrics
    volatility_metrics = metrics_analyzer.calculate_volatility(stock_df)
    
    # Calculate short squeeze metrics
    squeeze_metrics_df = metrics_analyzer.calculate_short_squeeze_metrics(short_interest_df, stock_df, options_df)
    
    if not squeeze_metrics_df.empty:
        squeeze_metrics_df.to_csv(os.path.join(output_dir, 'squeeze_metrics.csv'), index=False)
    
    # Step 5: Score and rank tickers
    logger.info("Step 5: Scoring and ranking potential squeeze candidates")
    scorer = SqueezeScorer()
    results_df = scorer.calculate_combined_score(ticker_mentions, ticker_sentiment, squeeze_metrics_df)
    
    # Save results
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'squeeze_results.csv'), index=False)
    
    # Step 6: Visualize results
    logger.info("Step 6: Visualizing results")
    visualizer = SqueezeVisualizer()
    
    if not results_df.empty:
        visualizer.save_visualizations(results_df, output_dir)
        
        # Display top squeeze candidates
        top_candidates = results_df[results_df['squeeze_candidate'] == True].sort_values('combined_score', ascending=False)
        
        if not top_candidates.empty:
            logger.info("\n===== TOP POTENTIAL SHORT SQUEEZE CANDIDATES =====")
            for _, row in top_candidates.head(5).iterrows():
                logger.info(f"{row['symbol']}: Combined Score {row['combined_score']:.2f}, "
                          f"Mentions: {row['mentions']}, "
                          f"Sentiment: {row['sentiment']:.2f}, "
                          f"Short Interest: {row['short_interest_percent'] if pd.notna(row['short_interest_percent']) else 'N/A'}%, "
                          f"Days to Cover: {row['days_to_cover'] if pd.notna(row['days_to_cover']) else 'N/A'}")
        else:
            logger.info("No potential short squeeze candidates found in this run")
        
        if dashboard:
            logger.info("Launching interactive dashboard")
            visualizer.create_dashboard(results_df)
    else:
        logger.warning("No results to visualize. Check error logs.")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    logger.info(f"All results saved to {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="APExplorer - Short Squeeze Detector")
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch interactive dashboard')
    
    args = parser.parse_args()


    run_pipeline(output_dir=args.output, dashboard=args.dashboard)