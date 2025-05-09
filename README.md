# APExplorer

APExplorer is a tool that identifies potential short squeeze candidates by analyzing Reddit's WallStreetBets discussions combined with financial metrics.

## Features

- **Social Media Analysis**: Extracts ticker mentions and sentiment from Reddit's WallStreetBets
- **Financial Data Integration**: Collects short interest, options data, and price trends from multiple sources
- **Scoring System**: Combines social and financial signals to identify high-potential squeeze candidates
- **Interactive Dashboard**: Visualizes results with detailed metrics and comparisons
- **Multi-Source Data**: Uses Yahoo Finance, Twelvedata, and NASDAQ Data Link for comprehensive data

## How It Works

1. **Data Collection**: Scrapes WSB posts and comments to identify trending stocks
2. **Text Processing**: Uses NLP to extract tickers and analyze sentiment
3. **Financial Analysis**: Collects short interest data, options activity, and price trends
4. **Scoring Algorithm**: Combines multiple factors to rank potential squeeze candidates
5. **Visualization**: Provides interactive charts and tables of results

## Setup

### Prerequisites
- Python 3.8+
- Reddit API credentials (for scraping)
- Twelvedata API key (for financial data)
- NASDAQ Data Link API key (optional, for additional financial data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/APExplorer.git
cd APExplorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
REDDIT_USER_AGENT=APExplorer/1.0 (by /u/your_username)
TWELVEDATA_API_KEY=your_twelvedata_key
NASDAQ_DATA_LINK_API_KEY=your_nasdaq_key
```

## Usage

### Basic Run
```bash
python main.py
```

### With Dashboard
```bash
python main.py --dashboard
```

### Custom Output Directory
```bash
python main.py --output ./my_results
```


## Key Components

### Social Signal Analysis
- Identifies ticker mentions in WSB posts and comments
- Analyzes sentiment using a custom WSB-aware model
- Tracks mention frequency and sentiment trends

### Financial Metrics
- Short interest percentage
- Days to cover ratio
- Options activity (call/put ratio)
- Price and volume trends

### Scoring System
- Weights different factors based on historical importance
- Normalizes and combines social and financial signals
- Identifies tickers exceeding threshold scores

## Data Sources

- **Reddit API**: For collecting posts and comments from WallStreetBets
- **Yahoo Finance**: Stock data, options chains, and some short interest data
- **Twelvedata**: Additional stock metrics and financial information
- **NASDAQ Data Link**: Advanced short interest data (requires API key)

## Customization

Edit `config/config.py` to customize:
- Data collection parameters
- Scoring weights
- Alert thresholds
- API settings

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.

## License

MIT