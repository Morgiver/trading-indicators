"""Pytest fixtures for trading-indicators tests."""

import pytest
from datetime import datetime
import yfinance as yf
from trading_frame import TimeFrame, Candle


@pytest.fixture(scope="session")
def qqq_1m_data():
    """
    Download QQQ 1-minute data using yfinance.

    Returns:
        pandas.DataFrame: QQQ 1-minute OHLCV data
    """
    # Download 7 days of 1-minute data for QQQ
    ticker = yf.Ticker("QQQ")
    data = ticker.history(period="7d", interval="1m")
    return data


@pytest.fixture(scope="session")
def qqq_candles(qqq_1m_data):
    """
    Convert QQQ yfinance data to Candle objects.

    Args:
        qqq_1m_data: QQQ 1-minute DataFrame from yfinance

    Returns:
        list[Candle]: List of Candle objects
    """
    candles = []
    for index, row in qqq_1m_data.iterrows():
        candle = Candle(
            date=index.to_pydatetime(),
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row['Volume'])
        )
        candles.append(candle)
    return candles


@pytest.fixture
def timeframe_1m():
    """
    Create a 1-minute TimeFrame.

    Returns:
        TimeFrame: Empty 1-minute timeframe
    """
    return TimeFrame('1T', max_periods=500)


@pytest.fixture
def populated_frame(timeframe_1m, qqq_candles):
    """
    Create a TimeFrame populated with QQQ data.

    Args:
        timeframe_1m: Empty 1-minute timeframe
        qqq_candles: List of Candle objects

    Returns:
        TimeFrame: Populated timeframe with QQQ data
    """
    for candle in qqq_candles[:500]:  # Limit to 500 candles
        timeframe_1m.feed(candle)
    return timeframe_1m
