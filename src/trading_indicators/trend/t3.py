"""T3 (Triple Exponential Moving Average T3) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class T3(BaseIndicator):
    """
    Triple Exponential Moving Average (T3) indicator.

    T3 is a type of moving average developed by Tim Tillson that provides a
    smoother and more responsive alternative to traditional moving averages.
    It applies exponential smoothing six times (not three times as the name
    might suggest) to achieve better lag reduction while maintaining smoothness.

    The T3 uses a volume factor (vfactor) to control the balance between
    smoothness and responsiveness. It's considered superior to DEMA and TEMA
    in terms of lag reduction and smoothness.

    Formula: T3 applies EMA six times with adjustable volume factor
    - Volume Factor (vfactor): Controls responsiveness vs smoothness
    - Range: 0.0 (most smooth) to 1.0 (most responsive)
    - Default: 0.7 (balanced)

    Characteristics:
    - Extremely smooth while maintaining low lag
    - Better noise filtering than EMA, DEMA, TEMA
    - Configurable responsiveness via volume factor
    - Minimal overshoot compared to other MAs
    - Common periods: 5, 8, 21
    - Common vfactor: 0.7 (can range 0.0-1.0)

    Usage:
    - Price above T3: Uptrend
    - Price below T3: Downtrend
    - T3 crossovers: Trading signals
      * Fast T3 crosses above Slow T3: Bullish signal
      * Fast T3 crosses below Slow T3: Bearish signal
    - Support/Resistance: Price often bounces off T3
    - Excellent for identifying trend without whipsaws

    Advantages:
    - Best lag-to-smoothness ratio
    - Excellent noise reduction
    - Minimal overshoot
    - Highly responsive with low false signals
    - Superior to DEMA and TEMA in most cases

    Volume Factor Guide:
    - 0.0: Maximum smoothness (similar to SMA)
    - 0.7: Balanced (default, recommended)
    - 1.0: Maximum responsiveness (more like EMA)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> t3_fast = T3(frame=frame, period=5, vfactor=0.7, column_name='T3_5')
        >>> t3_slow = T3(frame=frame, period=21, vfactor=0.7, column_name='T3_21')
        >>>
        >>> # Feed candles - T3 automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(f"T3(5): {t3_fast.periods[-1].T3_5}")
        >>> print(f"T3(21): {t3_slow.periods[-1].T3_21}")
        >>>
        >>> # Detect crossovers
        >>> if t3_fast.get_latest() > t3_slow.get_latest():
        ...     print("Fast T3 above Slow T3 - Bullish")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 5,
        vfactor: float = 0.7,
        column_name: str = 'T3',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize T3 indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for T3 calculation (default: 5)
                   Common values: 5, 8, 21
            vfactor: Volume factor (default: 0.7)
                    Range: 0.0 (smooth) to 1.0 (responsive)
                    Recommended: 0.7 (balanced)
            column_name: Name for the indicator column (default: 'T3')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2 or vfactor not in valid range
        """
        if period < 2:
            raise ValueError("T3 period must be at least 2")

        if not (0.0 <= vfactor <= 1.0):
            raise ValueError("T3 vfactor must be between 0.0 and 1.0")

        self.period = period
        self.vfactor = vfactor
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate T3 value for a specific period.

        Args:
            period: IndicatorPeriod to populate with T3 value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # T3 needs more periods due to multiple smoothing passes
        # Approximately 6 * period for stable results
        min_periods = self.period * 6
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

        # Calculate T3 using TA-Lib
        t3_values = talib.T3(prices_array, timeperiod=self.period, vfactor=self.vfactor)

        # The last value is the T3 for our period
        t3_value = t3_values[-1]

        if not np.isnan(t3_value):
            setattr(period, self.column_name, round(float(t3_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export T3 values as numpy array.

        Returns:
            NumPy array with T3 values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest T3 value.

        Returns:
            Latest T3 value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
