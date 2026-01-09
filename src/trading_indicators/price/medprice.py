"""MEDPRICE (Median Price) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MEDPRICE(BaseIndicator):
    """
    MEDPRICE (Median Price) indicator.

    Calculates the median (midpoint) price of a bar using High and Low.
    Formula: (High + Low) / 2

    This is useful for:
    - Identifying the middle of the trading range
    - Less sensitive to opening gaps than AVGPRICE
    - Common reference point for pivot calculations
    - Base for other technical indicators

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> medprice = MEDPRICE(frame=frame, column_name='MEDPRICE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(medprice.periods[-1].MEDPRICE)
        >>> print(medprice.get_latest())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'MEDPRICE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MEDPRICE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'MEDPRICE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MEDPRICE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with MEDPRICE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None:
            return

        # Extract High and Low prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate MEDPRICE using TA-Lib
        medprice_values = talib.MEDPRICE(high_prices, low_prices)

        # The last value is the MEDPRICE for our period
        medprice_value = medprice_values[-1]

        if not np.isnan(medprice_value):
            setattr(period, self.column_name, float(medprice_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export MEDPRICE values as numpy array.

        Returns:
            NumPy array with MEDPRICE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest MEDPRICE value.

        Returns:
            Latest MEDPRICE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_above_close(self) -> bool:
        """
        Check if median price is above close price.

        Returns:
            True if MEDPRICE > Close (close in lower half of range)
        """
        if self.periods:
            medprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if medprice is not None:
                return medprice > close
        return False

    def is_below_close(self) -> bool:
        """
        Check if median price is below close price.

        Returns:
            True if MEDPRICE < Close (close in upper half of range)
        """
        if self.periods:
            medprice = self.get_latest()
            close = self.frame.periods[-1].close_price
            if medprice is not None:
                return medprice < close
        return False

    def get_close_position_in_range(self) -> Optional[float]:
        """
        Get where the close is relative to the range (0 = low, 1 = high).

        Returns:
            Value between 0 and 1, or None if not available
        """
        if self.periods:
            period = self.frame.periods[-1]
            high = period.high_price
            low = period.low_price
            close = period.close_price

            if high != low:
                return (close - low) / (high - low)
        return None

    def get_range_size(self) -> Optional[float]:
        """
        Get the size of the high-low range.

        Returns:
            High - Low, or None if not available
        """
        if self.periods:
            period = self.frame.periods[-1]
            return period.high_price - period.low_price
        return None
