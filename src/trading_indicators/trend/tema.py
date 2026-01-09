"""TEMA (Triple Exponential Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class TEMA(BaseIndicator):
    """
    Triple Exponential Moving Average (TEMA) indicator.

    TEMA is a composite indicator that applies triple exponential smoothing
    to reduce lag even further than DEMA. It was developed by Patrick Mulloy
    and provides a more responsive moving average while maintaining smoothness.

    Formula: TEMA = 3 × EMA(n) - 3 × EMA(EMA(n)) + EMA(EMA(EMA(n)))
    where n is the time period

    The formula combines three EMAs in a way that significantly reduces lag
    while filtering out short-term noise.

    Characteristics:
    - Very low lag compared to standard EMA
    - More responsive than both EMA and DEMA
    - Smoother than simple price action
    - Reduces lag by using triple smoothing technique
    - Common periods: 9, 21, 50

    Usage:
    - Price above TEMA: Uptrend
    - Price below TEMA: Downtrend
    - TEMA crossovers: Trading signals
      * Fast TEMA crosses above Slow TEMA: Bullish signal
      * Fast TEMA crosses below Slow TEMA: Bearish signal
    - Support/Resistance: Price often bounces off TEMA levels
    - Excellent for fast-moving markets

    Advantages:
    - Extremely responsive to price changes
    - Maintains good smoothing despite low lag
    - Better than EMA and DEMA for trend following
    - Excellent for short-term trading
    - Fewer false signals than simple EMA

    Comparison:
    - SMA: Smoothest, most lag
    - EMA: Less lag than SMA
    - DEMA: Less lag than EMA
    - TEMA: Less lag than DEMA (most responsive)
    - T3: Similar responsiveness but smoother than TEMA

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> tema9 = TEMA(frame=frame, period=9, column_name='TEMA_9')
        >>> tema21 = TEMA(frame=frame, period=21, column_name='TEMA_21')
        >>>
        >>> # Feed candles - TEMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(f"TEMA(9): {tema9.periods[-1].TEMA_9}")
        >>> print(f"TEMA(21): {tema21.periods[-1].TEMA_21}")
        >>>
        >>> # Detect trend
        >>> if tema9.get_latest() > tema21.get_latest():
        ...     print("Fast TEMA above Slow TEMA - Bullish trend")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 21,
        column_name: str = 'TEMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize TEMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for TEMA calculation (default: 21)
                   Common values: 9, 21, 50
            column_name: Name for the indicator column (default: 'TEMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2
        """
        if period < 2:
            raise ValueError("TEMA period must be at least 2")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate TEMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with TEMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need enough periods for TEMA calculation (approximately 3 * period)
        min_periods = self.period * 3
        if period_index is None or period_index < min_periods - 1:
            return

        # Extract prices according to the specified field
        if self.price_field == 'close':
            prices = [p.close_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'high':
            prices = [p.high_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'low':
            prices = [p.low_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'open':
            prices = [p.open_price for p in self.frame.periods[:period_index + 1]]
        else:
            return

        prices_array = np.array(prices)

        # Remove NaN values
        prices_array = prices_array[~np.isnan(prices_array)]

        if len(prices_array) < min_periods:
            return

        # Calculate TEMA using TA-Lib
        tema_values = talib.TEMA(prices_array, timeperiod=self.period)

        # The last value is the TEMA for our period
        tema_value = tema_values[-1]

        if not np.isnan(tema_value):
            setattr(period, self.column_name, round(float(tema_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export TEMA values as numpy array.

        Returns:
            NumPy array with TEMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest TEMA value.

        Returns:
            Latest TEMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
