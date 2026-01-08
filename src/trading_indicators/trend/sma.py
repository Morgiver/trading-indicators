"""SMA (Simple Moving Average) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class SMA(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.

    SMA calculates the arithmetic mean of prices over a specified period.
    It's used to identify trends and smooth out price fluctuations.

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> sma20 = SMA(frame=frame, period=20, column_name='SMA_20')
        >>> sma50 = SMA(frame=frame, period=50, column_name='SMA_50')
        >>>
        >>> # Feed candles - SMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(sma20.periods[-1].SMA_20)
        >>> print(sma50.periods[-1].SMA_50)
        >>>
        >>> # Detect crossovers
        >>> if sma20.get_latest() > sma50.get_latest():
        ...     print("Golden cross (bullish)")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        column_name: str = 'SMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize SMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for SMA calculation (default: 20)
            column_name: Name for the indicator column (default: 'SMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate SMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with SMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for SMA calculation
        if period_index is None or period_index < self.period - 1:
            return

        # Extract prices according to the specified field
        if self.price_field == 'close':
            prices = [p.close_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'high':
            prices = [p.high_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'low':
            prices = [p.low_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'open':
            prices = [p.open_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        else:
            return

        prices_array = np.array(prices)

        # Calculate SMA using TA-Lib
        sma_values = talib.SMA(prices_array, timeperiod=self.period)

        # The last value is the SMA for our period
        sma_value = sma_values[-1]

        if not np.isnan(sma_value):
            setattr(period, self.column_name, float(sma_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export SMA values as numpy array.

        Returns:
            NumPy array with SMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest SMA value.

        Returns:
            Latest SMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
