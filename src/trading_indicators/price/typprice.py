"""TYPPRICE (Typical Price) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class TYPPRICE(BaseIndicator):
    """
    TYPPRICE (Typical Price) indicator.

    Calculates the typical price of a bar using High, Low, and Close.
    Formula: (High + Low + Close) / 3

    This is useful for:
    - Representing the "typical" price level of the bar
    - More weight on the close than MEDPRICE
    - Common input for volume-weighted indicators (e.g., VWAP)
    - Pivot point calculations

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> typprice = TYPPRICE(frame=frame, column_name='TYPPRICE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(typprice.periods[-1].TYPPRICE)
        >>> print(typprice.get_latest())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'TYPPRICE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize TYPPRICE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'TYPPRICE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate TYPPRICE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with TYPPRICE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None:
            return

        # Extract High, Low, and Close prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate TYPPRICE using TA-Lib
        typprice_values = talib.TYPPRICE(high_prices, low_prices, close_prices)

        # The last value is the TYPPRICE for our period
        typprice_value = typprice_values[-1]

        if not np.isnan(typprice_value):
            setattr(period, self.column_name, float(typprice_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export TYPPRICE values as numpy array.

        Returns:
            NumPy array with TYPPRICE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest TYPPRICE value.

        Returns:
            Latest TYPPRICE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_above_close(self) -> bool:
        """
        Check if typical price is above close price.

        Returns:
            True if TYPPRICE > Close (bearish bias)
        """
        if self.periods:
            typprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if typprice is not None:
                return typprice > close
        return False

    def is_below_close(self) -> bool:
        """
        Check if typical price is below close price.

        Returns:
            True if TYPPRICE < Close (bullish bias)
        """
        if self.periods:
            typprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if typprice is not None:
                return typprice < close
        return False

    def get_spread_from_close(self) -> Optional[float]:
        """
        Get the difference between TYPPRICE and Close.

        Returns:
            TYPPRICE - Close, or None if not available
        """
        if self.periods:
            typprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if typprice is not None:
                return typprice - close
        return None

    def is_rising(self, lookback: int = 1) -> bool:
        """
        Check if typical price is rising.

        Args:
            lookback: Number of periods to look back (default: 1)

        Returns:
            True if current TYPPRICE > previous TYPPRICE
        """
        if len(self.periods) >= lookback + 1:
            current = getattr(self.periods[-1], self.column_name, None)
            previous = getattr(self.periods[-lookback - 1], self.column_name, None)

            if current is not None and previous is not None:
                return current > previous
        return False

    def is_falling(self, lookback: int = 1) -> bool:
        """
        Check if typical price is falling.

        Args:
            lookback: Number of periods to look back (default: 1)

        Returns:
            True if current TYPPRICE < previous TYPPRICE
        """
        if len(self.periods) >= lookback + 1:
            current = getattr(self.periods[-1], self.column_name, None)
            previous = getattr(self.periods[-lookback - 1], self.column_name, None)

            if current is not None and previous is not None:
                return current < previous
        return False
