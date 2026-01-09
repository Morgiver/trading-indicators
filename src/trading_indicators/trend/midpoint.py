"""MIDPOINT (MidPoint over period) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class MIDPOINT(BaseIndicator):
    """
    MidPoint over period indicator.

    MIDPOINT calculates the midpoint (middle value) of the data over a specified
    time period. It's essentially (highest value + lowest value) / 2 over the
    period. This provides a simple measure of the central tendency of price
    action over the lookback period.

    Formula: MIDPOINT = (MAX(price, period) + MIN(price, period)) / 2

    Characteristics:
    - Simple measure of price center over period
    - Similar to moving average but uses max/min instead of average
    - Less smooth than MA, more reactive to extremes
    - Provides a dynamic median line
    - Common periods: 14, 20, 50

    Usage:
    - Price above MIDPOINT: Bullish bias
    - Price below MIDPOINT: Bearish bias
    - MIDPOINT as support/resistance: Price often bounces off this level
    - Breakout detection: Price moving away from MIDPOINT
    - Mean reversion: Price returning to MIDPOINT

    Comparison with other indicators:
    - Simpler than MIDPRICE (which uses high/low specifically)
    - More reactive than SMA to price extremes
    - Less smooth than EMA
    - Useful for range-bound markets

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> midpoint = MIDPOINT(frame=frame, period=14, column_name='MIDPOINT_14')
        >>>
        >>> # Feed candles - MIDPOINT automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(midpoint.periods[-1].MIDPOINT_14)
        >>>
        >>> # Use as support/resistance
        >>> current_price = frame.periods[-1].close_price
        >>> midpoint_value = midpoint.get_latest()
        >>> if current_price > midpoint_value:
        ...     print("Price above midpoint - bullish")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 14,
        column_name: str = 'MIDPOINT',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MIDPOINT indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for MIDPOINT calculation (default: 14)
                   Common values: 14, 20, 50
            column_name: Name for the indicator column (default: 'MIDPOINT')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2
        """
        if period < 2:
            raise ValueError("MIDPOINT period must be at least 2")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MIDPOINT value for a specific period.

        Args:
            period: IndicatorPeriod to populate with MIDPOINT value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for MIDPOINT calculation
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

        # Remove NaN values
        prices_array = prices_array[~np.isnan(prices_array)]

        if len(prices_array) < self.period:
            return

        # Calculate MIDPOINT using TA-Lib
        midpoint_values = talib.MIDPOINT(prices_array, timeperiod=self.period)

        # The last value is the MIDPOINT for our period
        midpoint_value = midpoint_values[-1]

        if not np.isnan(midpoint_value):
            setattr(period, self.column_name, round(float(midpoint_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export MIDPOINT values as numpy array.

        Returns:
            NumPy array with MIDPOINT values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest MIDPOINT value.

        Returns:
            Latest MIDPOINT value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
