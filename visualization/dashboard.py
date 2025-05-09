import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SqueezeVisualizer:
    def __init__(self):
        self.colors = {
            'background': '#f8f9fa',
            'text': '#343a40',
            'green': '#28a745',
            'red': '#dc3545',
            'blue': '#007bff',
            'yellow': '#ffc107',
            'purple': '#6f42c1'
        }
    
    def plot_top_mentioned(self, results_df, top_n=10):
        """
        Plot top mentioned tickers
        """
        if results_df.empty:
            return None
            
        # Get top N tickers by mentions
        top_df = results_df.sort_values('mentions', ascending=False).head(top_n)
        
        # Create bar chart
        fig = px.bar(
            top_df,
            x='symbol',
            y='mentions',
            color='sentiment',
            color_continuous_scale=['red', 'gray', 'green'],
            range_color=[-1, 1],
            title=f'Top {top_n} Mentioned Tickers with Sentiment',
            labels={'mentions': 'Number of Mentions', 'symbol': 'Ticker', 'sentiment': 'Sentiment Score'},
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text']
        )
        
        return fig
    
    def plot_squeeze_candidates(self, results_df):
        """
        Plot squeeze candidate metrics
        """
        if results_df.empty:
            return None
            
        # Filter to squeeze candidates
        candidates = results_df[results_df['squeeze_candidate'] == True].sort_values('combined_score', ascending=False)
        
        if candidates.empty:
            return None
            
        # Create scatter plot
        fig = px.scatter(
            candidates,
            x='short_interest_percent',
            y='mentions',
            size='combined_score',
            color='sentiment',
            hover_name='symbol',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[-1, 1],
            title='Potential Short Squeeze Candidates',
            labels={
                'short_interest_percent': 'Short Interest %', 
                'mentions': 'Mentions Count',
                'sentiment': 'Sentiment Score'
            },
            height=600
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text']
        )
        
        return fig
    
    def plot_score_comparison(self, results_df, top_n=10):
        """
        Plot score comparison for top tickers
        """
        if results_df.empty:
            return None
            
        top_df = results_df.sort_values('combined_score', ascending=False).head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_df['symbol'],
            y=top_df['social_score'],
            name='Social Score',
            marker_color=self.colors['blue']
        ))
        
        fig.add_trace(go.Bar(
            x=top_df['symbol'],
            y=top_df['financial_score'],
            name='Financial Score',
            marker_color=self.colors['green']
        ))
        
        fig.add_trace(go.Bar(
            x=top_df['symbol'],
            y=top_df['combined_score'],
            name='Combined Score',
            marker_color=self.colors['purple']
        ))
        
        fig.update_layout(
            title=f'Score Comparison for Top {top_n} Tickers',
            xaxis_title='Ticker',
            yaxis_title='Score (0-1)',
            barmode='group',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font_color=self.colors['text'],
            height=500
        )
        
        return fig
    
    def create_dashboard(self, results_df):
        """
        Create a Dash dashboard to visualize results
        """
        # Initialize the Dash app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div(style={'backgroundColor': self.colors['background'], 'padding': '20px'}, children=[
            html.H1(
                'APExplorer - Short Squeeze Detector Dashboard',
                style={'textAlign': 'center', 'color': self.colors['text']}
            ),
            
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px 0px', 'borderRadius': '5px'}, children=[
                html.H2('Potential Squeeze Candidates', style={'color': self.colors['text']}),
                
                html.Div([
                    html.Table(
                        # Header
                        [html.Tr([
                            html.Th('Ticker'),
                            html.Th('Combined Score'),
                            html.Th('Mentions'),
                            html.Th('Sentiment'),
                            html.Th('Short Interest %'),
                            html.Th('Days to Cover')
                        ])] +
                        # Body
                        [html.Tr([
                            html.Td(row['symbol']),
                            html.Td(f"{row['combined_score']:.2f}"),
                            html.Td(row['mentions']),
                            html.Td(f"{row['sentiment']:.2f}"),
                            html.Td(f"{row['short_interest_percent']:.2f}" if pd.notna(row['short_interest_percent']) else 'N/A'),
                            html.Td(f"{row['days_to_cover']:.2f}" if pd.notna(row['days_to_cover']) else 'N/A')
                        ]) for _, row in results_df[results_df['squeeze_candidate'] == True].sort_values('combined_score', ascending=False).iterrows()],
                        style={'width': '100%', 'textAlign': 'center'}
                    )
                ])
            ]),
            
            # Top mentioned tickers
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px 0px', 'borderRadius': '5px'}, children=[
                html.H2('Top Mentioned Tickers', style={'color': self.colors['text']}),
                dcc.Graph(
                    id='mentions-graph',
                    figure=self.plot_top_mentioned(results_df)
                )
            ]),
            
            # Squeeze candidates plot
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px 0px', 'borderRadius': '5px'}, children=[
                html.H2('Squeeze Candidates Visualization', style={'color': self.colors['text']}),
                dcc.Graph(
                    id='candidates-graph',
                    figure=self.plot_squeeze_candidates(results_df)
                )
            ]),
            
            # Score comparison
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px 0px', 'borderRadius': '5px'}, children=[
                html.H2('Score Comparison', style={'color': self.colors['text']}),
                dcc.Graph(
                    id='score-graph',
                    figure=self.plot_score_comparison(results_df)
                )
            ]),
            
            html.Div(style={'textAlign': 'center', 'margin': '20px 0px', 'color': self.colors['text']}, children=[
                html.P(f"Last Updated: {results_df['timestamp'].iloc[0] if not results_df.empty else 'N/A'}")
            ])
        ])
        
        # Run the app
        logger.info("Starting dashboard server")
        app.run_server(debug=True)
    
    def save_visualizations(self, results_df, output_dir='./output'):
        """
        Save visualizations to files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        top_mentioned_fig = self.plot_top_mentioned(results_df)
        squeeze_candidates_fig = self.plot_squeeze_candidates(results_df)
        score_comparison_fig = self.plot_score_comparison(results_df)
        
        if top_mentioned_fig:
            top_mentioned_fig.write_html(os.path.join(output_dir, 'top_mentioned.html'))
            
        if squeeze_candidates_fig:
            squeeze_candidates_fig.write_html(os.path.join(output_dir, 'squeeze_candidates.html'))
            
        if score_comparison_fig:
            score_comparison_fig.write_html(os.path.join(output_dir, 'score_comparison.html'))
        
        results_df.to_csv(os.path.join(output_dir, 'squeeze_results.csv'), index=False)
        
        logger.info(f"Saved visualizations to {output_dir}")