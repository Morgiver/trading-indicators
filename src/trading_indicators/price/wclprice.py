"""WCLPRICE (Weighted Close Price) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class WCLPRICE(BaseIndicator):
    """
    WCLPRICE (Weighted Close Price) indicator.

    Calculates the weighted close price with emphasis on the close.
    Formula: (High + Low + Close + Close) / 4 = (High + Low + 2*Close) / 4

    This is useful for:
    - Giving more weight to the close price (2x)
    - Similar to AVGPRICE but emphasizes where the bar closed
    - Smoother than using only close
    - Better represents the final market sentiment

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> wclprice = WCLPRICE(frame=frame, column_name='WCLPRICE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(wclprice.periods[-1].WCLPRICE)
        >>> print(wclprice.get_latest())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'WCLPRICE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize WCLPRICE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'WCLPRICE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate WCLPRICE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with WCLPRICE value
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

        # Calculate WCLPRICE using TA-Lib
        wclprice_values = talib.WCLPRICE(high_prices, low_prices, close_prices)

        # The last value is the WCLPRICE for our period
        wclprice_value = wclprice_values[-1]

        if not np.isnan(wclprice_value):
            setattr(period, self.column_name, float(wclprice_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export WCLPRICE values as numpy array.

        Returns:
            NumPy array with WCLPRICE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest WCLPRICE value.

        Returns:
            Latest WCLPRICE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_above_close(self) -> bool:
        """
        Check if weighted close price is above actual close price.

        Returns:
            True if WCLPRICE > Close (unusual, means high is very high)
        """
        if self.periods:
            wclprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if wclprice is not None:
                return wclprice > close
        return False

    def is_below_close(self) -> bool:
        """
        Check if weighted close price is below actual close price.

        Returns:
            True if WCLPRICE < Close (unusual, means low is very low)
        """
        if self.periods:
            wclprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if wclprice is not None:
                return wclprice < close
        return False

    def get_spread_from_close(self) -> Optional[float]:
        """
        Get the difference between WCLPRICE and Close.

        Returns:
            WCLPRICE - Close, or None if not available
        """
        if self.periods:
            wclprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if wclprice is not None:
                return wclprice - close
        return None

    def is_rising(self, lookback: int = 1) -> bool:
        """
        Check if weighted close price is rising.

        Args:
            lookback: Number of periods to look back (default: 1)

        Returns:
            True if current WCLPRICE > previous WCLPRICE
        """
        if len(self.periods) >= lookback + 1:
            current = getattr(self.periods[-1], self.column_name, None)
            previous = getattr(self.periods[-lookback - 1], self.column_name, None)

            if current is not None and previous is not None:
                return current > previous
        return False

    def is_falling(self, lookback: int = 1) -> bool:
        """
        Check if weighted close price is falling.

        Args:
            lookback: Number of periods to look back (default: 1)

        Returns:
            True if current WCLPRICE < previous WCLPRICE
        """
        if len(self.periods) >= lookback + 1:
            current = getattr(self.periods[-1], self.column_name, None)
            previous = getattr(self.periods[-lookback - 1], self.column_name, None)

            if current is not None and previous is not None:
                return current < previous
        return False

    def get_momentum(self, lookback: int = 1) -> Optional[float]:
        """
        Get the momentum (change) of weighted close price.

        Args:
            lookback: Number of periods to look back (default: 1)

        Returns:
            Current WCLPRICE - Previous WCLPRICE, or None if not available
        """
        if len(self.periods) >= lookback + 1:
            current = getattr(self.periods[-1], self.column_name, None)
            previous = getattr(self.periods[-lookback - 1], self.column_name, None)

            if current is not None and previous is not None:
                return current - previous
        return None
