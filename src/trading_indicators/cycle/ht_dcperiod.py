"""HT_DCPERIOD (Hilbert Transform - Dominant Cycle Period) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class HT_DCPERIOD(BaseIndicator):
    """
    HT_DCPERIOD (Hilbert Transform - Dominant Cycle Period) indicator.

    Uses Hilbert Transform to identify the dominant cycle period in the market.
    This helps traders understand the current market cycle length, which is useful
    for adaptive trading strategies and cycle-based analysis.

    The dominant cycle period represents the number of bars/periods in the
    current dominant market cycle.

    Typical interpretation:
    - Short periods (< 20): Fast cycles, more volatile market
    - Medium periods (20-40): Normal cycling behavior
    - Long periods (> 40): Slow cycles, trending market
    - Changing periods: Market transitioning between cycle modes

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ht_dcperiod = HT_DCPERIOD(frame=frame, column_name='HT_DCPERIOD')
        >>>
        >>> # Feed candles - indicator automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ht_dcperiod.periods[-1].HT_DCPERIOD)
        >>> print(ht_dcperiod.get_cycle_type())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'HT_DCPERIOD',
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_DCPERIOD indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'HT_DCPERIOD')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_DCPERIOD value for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_DCPERIOD value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 32 periods for HT_DCPERIOD calculation (Hilbert Transform requirement)
        if period_index is None or len(self.frame.periods) < 32:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate HT_DCPERIOD using TA-Lib
        ht_dcperiod_values = talib.HT_DCPERIOD(close_prices)

        # The last value is the HT_DCPERIOD for our period
        ht_dcperiod_value = ht_dcperiod_values[-1]

        if not np.isnan(ht_dcperiod_value):
            setattr(period, self.column_name, float(ht_dcperiod_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export HT_DCPERIOD values as numpy array.

        Returns:
            NumPy array with HT_DCPERIOD values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest HT_DCPERIOD value.

        Returns:
            Latest HT_DCPERIOD value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def get_cycle_type(self) -> Optional[str]:
        """
        Classify the current cycle type based on period length.

        Returns:
            'fast', 'normal', 'slow', or None if not available
        """
        latest = self.get_latest()
        if latest is None:
            return None

        if latest < 20:
            return 'fast'
        elif latest <= 40:
            return 'normal'
        else:
            return 'slow'

    def is_cycle_shortening(self, lookback: int = 5) -> bool:
        """
        Check if cycle period is getting shorter.

        Args:
            lookback: Number of periods to look back (default: 5)

        Returns:
            True if cycle is shortening
        """
        if len(self.periods) < lookback:
            return False

        old_period = getattr(self.periods[-lookback], self.column_name, None)
        curr_period = getattr(self.periods[-1], self.column_name, None)

        if old_period is not None and curr_period is not None:
            return curr_period < old_period

        return False

    def is_cycle_lengthening(self, lookback: int = 5) -> bool:
        """
        Check if cycle period is getting longer.

        Args:
            lookback: Number of periods to look back (default: 5)

        Returns:
            True if cycle is lengthening
        """
        if len(self.periods) < lookback:
            return False

        old_period = getattr(self.periods[-lookback], self.column_name, None)
        curr_period = getattr(self.periods[-1], self.column_name, None)

        if old_period is not None and curr_period is not None:
            return curr_period > old_period

        return False

    def get_average_cycle(self, lookback: int = 20) -> Optional[float]:
        """
        Calculate average cycle period over lookback periods.

        Args:
            lookback: Number of periods to average (default: 20)

        Returns:
            Average cycle period or None
        """
        if len(self.periods) < lookback:
            return None

        values = [
            getattr(p, self.column_name, None)
            for p in self.periods[-lookback:]
        ]
        valid_values = [v for v in values if v is not None]

        if valid_values:
            return sum(valid_values) / len(valid_values)

        return None
