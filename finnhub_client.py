import finnhub
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Tuple

# Load environment variables from .env file in the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class FinnhubClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client.
        
        Args:
            api_key: Finnhub API key. If None, looks for FINNHUB_API_KEY in environment.
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API key not found. Please provide it or set FINNHUB_API_KEY in .env file.")
        
        self.client = finnhub.Client(api_key=self.api_key)

    def get_historical_data(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            resolution: Resolution (1, 5, 15, 30, 60, D, W, M)
            days_back: Number of days of historical data to fetch

        Returns:
            DataFrame with columns 'ds' and 'y' (close price)
        """
        end_time = int(time.time())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())

        try:
            res = self.client.stock_candles(symbol, resolution, start_time, end_time)
        except Exception as e:
            if '403' in str(e):
                raise Exception(
                    f"Access denied to historical data. Your Finnhub API tier does not have access to the stock_candles endpoint.\n\n"
                    f"To use historical data fetching, upgrade to Finnhub Pro+ tier at https://finnhub.io/pricing\n\n"
                    f"Error: {str(e)}"
                )
            raise

        if res['s'] == 'no_data':
            raise ValueError(f"No data found for symbol {symbol} with resolution {resolution}")

        if res['s'] != 'ok':
            raise Exception(f"Error fetching data from Finnhub: {res.get('msg', 'Unknown error')}")

        df = pd.DataFrame({
            'ds': pd.to_datetime(res['t'], unit='s'),
            'y': res['c']  # Use close price
        })

        return df

def fetch_stock_data(symbol: str, resolution: str = 'D', days_back: int = 365) -> pd.DataFrame:
    """Helper function to fetch stock data using environment API key"""
    client = FinnhubClient()
    return client.get_historical_data(symbol, resolution, days_back)
