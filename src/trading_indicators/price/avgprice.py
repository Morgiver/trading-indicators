"""AVGPRICE (Average Price) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class AVGPRICE(BaseIndicator):
    """
    AVGPRICE (Average Price) indicator.

    Calculates the average price of a bar using Open, High, Low, and Close.
    Formula: (Open + High + Low + Close) / 4

    This is useful for:
    - Smoothing price action
    - Representing the "center" of the bar
    - Reducing noise compared to using only Close
    - Base for other calculations

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> avgprice = AVGPRICE(frame=frame, column_name='AVGPRICE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(avgprice.periods[-1].AVGPRICE)
        >>> print(avgprice.get_latest())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'AVGPRICE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize AVGPRICE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'AVGPRICE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate AVGPRICE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with AVGPRICE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None:
            return

        # Extract OHLC prices
        open_prices = np.array([p.open_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate AVGPRICE using TA-Lib
        avgprice_values = talib.AVGPRICE(open_prices, high_prices, low_prices, close_prices)

        # The last value is the AVGPRICE for our period
        avgprice_value = avgprice_values[-1]

        if not np.isnan(avgprice_value):
            setattr(period, self.column_name, float(avgprice_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export AVGPRICE values as numpy array.

        Returns:
            NumPy array with AVGPRICE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest AVGPRICE value.

        Returns:
            Latest AVGPRICE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_above_close(self) -> bool:
        """
        Check if average price is above close price (bearish bar).

        Returns:
            True if AVGPRICE > Close (more weight on high/open)
        """
        if self.frame.periods:
            avgprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if avgprice is not None:
                return avgprice > close
        return False

    def is_below_close(self) -> bool:
        """
        Check if average price is below close price (bullish bar).

        Returns:
            True if AVGPRICE < Close (more weight on low/open)
        """
        if self.frame.periods:
            avgprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if avgprice is not None:
                return avgprice < close
        return False

    def get_spread_from_close(self) -> Optional[float]:
        """
        Get the difference between AVGPRICE and Close.

        Returns:
            AVGPRICE - Close, or None if not available
        """
        if self.frame.periods:
            avgprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if avgprice is not None:
                return avgprice - close
        return None
