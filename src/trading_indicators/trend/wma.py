"""WMA (Weighted Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class WMA(BaseIndicator):
    """
    Weighted Moving Average (WMA) indicator.

    WMA assigns linearly increasing weights to more recent data points, making
    it more responsive to recent price changes compared to SMA. The most recent
    price gets weight n, the second most recent gets weight n-1, and so on down
    to the oldest price which gets weight 1.

    Formula: WMA = Σ(Price[i] × Weight[i]) / Σ(Weight[i])
    where Weight[i] = (n - i + 1) for i from 1 to n

    Example for WMA(5):
    - Price[0] (oldest) gets weight 1
    - Price[1] gets weight 2
    - Price[2] gets weight 3
    - Price[3] gets weight 4
    - Price[4] (newest) gets weight 5
    - WMA = (P[0]×1 + P[1]×2 + P[2]×3 + P[3]×4 + P[4]×5) / (1+2+3+4+5)
    - WMA = (P[0]×1 + P[1]×2 + P[2]×3 + P[3]×4 + P[4]×5) / 15

    Characteristics:
    - More weight on recent prices (linear progression)
    - Less lag than SMA
    - More responsive than SMA but less than EMA
    - Smoothness between SMA and EMA
    - Common periods: 10, 20, 50

    Usage:
    - Price above WMA: Uptrend
    - Price below WMA: Downtrend
    - WMA crossovers: Trading signals
      * Fast WMA crosses above Slow WMA: Bullish signal
      * Fast WMA crosses below Slow WMA: Bearish signal
    - Support/Resistance: Price often bounces off WMA levels
    - Balanced between smoothness and responsiveness

    Advantages:
    - More responsive than SMA
    - Simple and intuitive weighting scheme
    - Less lag than SMA
    - Balanced approach to trend following
    - Good middle ground between SMA and EMA

    Comparison:
    - SMA: All prices weighted equally (smoothest, most lag)
    - WMA: Linear increasing weights (middle ground)
    - EMA: Exponential weights (most responsive, least lag)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> wma10 = WMA(frame=frame, period=10, column_name='WMA_10')
        >>> wma20 = WMA(frame=frame, period=20, column_name='WMA_20')
        >>>
        >>> # Feed candles - WMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(f"WMA(10): {wma10.periods[-1].WMA_10}")
        >>> print(f"WMA(20): {wma20.periods[-1].WMA_20}")
        >>>
        >>> # Detect golden cross
        >>> if wma10.get_latest() > wma20.get_latest():
        ...     print("Fast WMA above Slow WMA - Bullish")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        column_name: str = 'WMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize WMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for WMA calculation (default: 20)
                   Common values: 10, 20, 50
            column_name: Name for the indicator column (default: 'WMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("WMA period must be at least 1")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate WMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with WMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for WMA calculation
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

        # Calculate WMA using TA-Lib
        wma_values = talib.WMA(prices_array, timeperiod=self.period)

        # The last value is the WMA for our period
        wma_value = wma_values[-1]

        if not np.isnan(wma_value):
            setattr(period, self.column_name, round(float(wma_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export WMA values as numpy array.

        Returns:
            NumPy array with WMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest WMA value.

        Returns:
            Latest WMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
