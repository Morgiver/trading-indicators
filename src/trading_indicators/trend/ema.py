"""EMA (Exponential Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.

    Weighted moving average that gives more importance to recent prices.
    More responsive to recent price changes than SMA.

    Formula: EMA = (Price - Previous EMA) × Multiplier + Previous EMA
    where Multiplier = 2 / (period + 1)

    Characteristics:
    - Less lag than SMA (more responsive to recent price changes)
    - More weight to recent prices, exponentially decreasing for older prices
    - More sensitive to price movements than SMA
    - Common periods: 9, 12, 20, 26, 50, 200

    Usage:
    - Price above EMA: Uptrend
    - Price below EMA: Downtrend
    - EMA crossovers: Trading signals
      * Fast EMA crosses above Slow EMA: Bullish signal
      * Fast EMA crosses below Slow EMA: Bearish signal
    - Support/Resistance: Price often bounces off EMA levels
    - Used in MACD (12 EMA - 26 EMA)

    Advantages over SMA:
    - Reacts faster to price changes
    - Better for short-term trading
    - Reduces lag in trend identification

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ema12 = EMA(frame=frame, period=12, column_name='EMA_12')
        >>> ema26 = EMA(frame=frame, period=26, column_name='EMA_26')
        >>>
        >>> # Feed candles - EMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ema12.periods[-1].EMA_12)
        >>> print(ema26.periods[-1].EMA_26)
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        column_name: str = 'EMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize EMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for EMA calculation (default: 20)
                   Common values: 9, 12, 20, 26, 50, 100, 200
            column_name: Name for the indicator column (default: 'EMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("EMA period must be at least 1")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate EMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with EMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for EMA calculation
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

        # Calculate EMA using TA-Lib
        ema_values = talib.EMA(prices_array, timeperiod=self.period)

        # The last value is the EMA for our period
        ema_value = ema_values[-1]

        if not np.isnan(ema_value):
            setattr(period, self.column_name, round(float(ema_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export EMA values as numpy array.

        Returns:
            NumPy array with EMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest EMA value.

        Returns:
            Latest EMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
