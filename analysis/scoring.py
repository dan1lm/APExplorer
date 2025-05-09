import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config.config import (
    WEIGHT_MENTIONS,
    WEIGHT_SENTIMENT,
    WEIGHT_SHORT_INTEREST,
    WEIGHT_DAYS_TO_COVER,
    WEIGHT_FLOAT_PERCENTAGE,
    SQUEEZE_SCORE_THRESHOLD
)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SqueezeScorer:
    def __init__(self):
        pass
    
    def normalize_mentions(self, ticker_mentions):
        """
        Normalization: mention counts to a 0-1 scale
        """
        if not ticker_mentions:
            return {}
            
        mentions = list(ticker_mentions.values())
        max_mentions = max(mentions)
        min_mentions = min(mentions)
        
        range_mentions = max_mentions - min_mentions
        
        if range_mentions == 0:
            return {ticker: 1.0 for ticker in ticker_mentions}
        
        normalized = {}
        for ticker, count in ticker_mentions.items():
            normalized[ticker] = (count - min_mentions) / range_mentions
            
        return normalized
    
    def normalize_sentiment(self, ticker_sentiment):
        """
        Normalization: sentiment scores to a 0-1 scale
        """
        if not ticker_sentiment:
            return {}
            
        # Convert compound sentiment (-1 to 1) to 0-1 scale
        normalized = {}
        for ticker, sentiment in ticker_sentiment.items():
            normalized[ticker] = (sentiment + 1) / 2
            
        return normalized
    
    def normalize_financial_metrics(self, squeeze_metrics_df):
        """
        Normalization: financial metrics to 0-1 scales
        """
        if squeeze_metrics_df.empty:
            return pd.DataFrame()
            
        normalized_df = squeeze_metrics_df.copy()
        
        # Normalization: short interest percentage
        if 'short_interest_percent' in normalized_df.columns:
            si_max = normalized_df['short_interest_percent'].max()
            si_min = normalized_df['short_interest_percent'].min()
            if si_max > si_min:
                normalized_df['norm_short_interest'] = (
                    (normalized_df['short_interest_percent'] - si_min) / (si_max - si_min)
                )
            else:
                normalized_df['norm_short_interest'] = 1.0
                
        # days to cover notmalization
        if 'days_to_cover' in normalized_df.columns:
            dtc_max = normalized_df['days_to_cover'].max()
            dtc_min = normalized_df['days_to_cover'].min()
            if dtc_max > dtc_min:
                normalized_df['norm_days_to_cover'] = (
                    (normalized_df['days_to_cover'] - dtc_min) / (dtc_max - dtc_min)
                )
            else:
                normalized_df['norm_days_to_cover'] = 1.0
                
        # Use squeeze_potential as is if it exists
        if 'squeeze_potential' in normalized_df.columns:
            normalized_df['norm_squeeze_potential'] = normalized_df['squeeze_potential']
            
        return normalized_df
    
    def calculate_combined_score(
        self, 
        ticker_mentions, 
        ticker_sentiment, 
        squeeze_metrics_df
    ):
        """
        Calculate combined score using social and financial metrics
        """
        # Normalize inputs
        norm_mentions = self.normalize_mentions(ticker_mentions)
        norm_sentiment = self.normalize_sentiment(ticker_sentiment)
        norm_metrics_df = self.normalize_financial_metrics(squeeze_metrics_df)
        
        # Create a set of all tickers
        all_tickers = set(norm_mentions.keys()) | set(norm_sentiment.keys())
        if not norm_metrics_df.empty:
            all_tickers |= set(norm_metrics_df['symbol'].unique())
        
        results = []
        
        for ticker in all_tickers:
            mention_score = norm_mentions.get(ticker, 0.0)
            sentiment_score = norm_sentiment.get(ticker, 0.5)  # Neutral if missing
        
            ticker_metrics = norm_metrics_df[norm_metrics_df['symbol'] == ticker] if not norm_metrics_df.empty else pd.DataFrame()
            
            si_score = ticker_metrics['norm_short_interest'].values[0] if not ticker_metrics.empty and 'norm_short_interest' in ticker_metrics.columns else 0.0
            dtc_score = ticker_metrics['norm_days_to_cover'].values[0] if not ticker_metrics.empty and 'norm_days_to_cover' in ticker_metrics.columns else 0.0
            squeeze_score = ticker_metrics['norm_squeeze_potential'].values[0] if not ticker_metrics.empty and 'norm_squeeze_potential' in ticker_metrics.columns else 0.0
            
            social_score = (
                WEIGHT_MENTIONS * mention_score + 
                WEIGHT_SENTIMENT * sentiment_score
            ) / (WEIGHT_MENTIONS + WEIGHT_SENTIMENT)

            financial_score = (
                WEIGHT_SHORT_INTEREST * si_score + 
                WEIGHT_DAYS_TO_COVER * dtc_score +
                WEIGHT_FLOAT_PERCENTAGE * squeeze_score
            ) / (WEIGHT_SHORT_INTEREST + WEIGHT_DAYS_TO_COVER + WEIGHT_FLOAT_PERCENTAGE)
            
            # Calculate final combined score weighted average
            combined_score = 0.6 * financial_score + 0.4 * social_score
            
            raw_mentions = ticker_mentions.get(ticker, 0)
            raw_sentiment = ticker_sentiment.get(ticker, 0)
            raw_si = ticker_metrics['short_interest_percent'].values[0] if not ticker_metrics.empty and 'short_interest_percent' in ticker_metrics.columns else None
            raw_dtc = ticker_metrics['days_to_cover'].values[0] if not ticker_metrics.empty and 'days_to_cover' in ticker_metrics.columns else None
            
            result = {
                'symbol': ticker,
                'mentions': raw_mentions,
                'sentiment': raw_sentiment,
                'short_interest_percent': raw_si,
                'days_to_cover': raw_dtc,
                'social_score': social_score,
                'financial_score': financial_score,
                'combined_score': combined_score,
                'squeeze_candidate': combined_score >= SQUEEZE_SCORE_THRESHOLD,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('combined_score', ascending=False)
        
        logger.info(f"Calculated combined scores for {len(results_df)} tickers")
        logger.info(f"Found {results_df['squeeze_candidate'].sum()} potential squeeze candidates")
        
        return results_df