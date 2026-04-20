import yfinance as yf
import pandas as pd
from typing import Optional

class YFinanceClient:
    """
    Free stock data client using yfinance.
    No API key required. Supports daily, weekly, and monthly historical data.
    """

    def __init__(self):
        """Initialize YFinance client (no API key needed)."""
        self.client = yf.Ticker

    def get_historical_data(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            resolution: Resolution ('D'=daily, 'W'=weekly, 'M'=monthly)
            days_back: Number of days of historical data to fetch

        Returns:
            DataFrame with columns 'ds' and 'y' (close price)
        """
        # yfinance doesn't support intraday without premium
        # Only support D, W, M
        valid_intervals = {'D': '1d', 'W': '1wk', 'M': '1mo'}

        if resolution not in valid_intervals:
            raise ValueError(
                f"yfinance (free tier) only supports daily/weekly/monthly.\n"
                f"Requested: {resolution}\n"
                f"Supported: D (daily), W (weekly), M (monthly)"
            )

        interval = valid_intervals[resolution]

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days_back}d", interval=interval)

            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Convert to our standard format
            df = pd.DataFrame({
                'ds': hist.index,
                'y': hist['Close'].values
            })

            # Reset index to ensure ds is a column, not index
            df = df.reset_index(drop=True)
            df['ds'] = pd.to_datetime(df['ds'])

            return df

        except Exception as e:
            if "No data" in str(e) or "invalid" in str(e).lower():
                raise ValueError(f"Invalid symbol or no data available: {symbol}")
            raise Exception(f"Error fetching data from Yahoo Finance: {str(e)}")


def fetch_stock_data(symbol: str, resolution: str = 'D', days_back: int = 365) -> pd.DataFrame:
    """
    Helper function to fetch stock data using yfinance (free, no API key required).

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        resolution: Resolution ('D'=daily, 'W'=weekly, 'M'=monthly)
        days_back: Number of days of historical data to fetch

    Returns:
        DataFrame with columns 'ds' and 'y' (close price)
    """
    client = YFinanceClient()
    return client.get_historical_data(symbol, resolution, days_back)
