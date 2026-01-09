"""HT_TRENDMODE (Hilbert Transform - Trend vs Cycle Mode) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class HT_TRENDMODE(BaseIndicator):
    """
    HT_TRENDMODE (Hilbert Transform - Trend vs Cycle Mode) indicator.

    Uses Hilbert Transform to determine if the market is in trend mode
    or cycle mode. Returns 1 for trend mode and 0 for cycle mode.

    This helps traders adapt their strategy:
    - Trend mode (1): Use trend-following strategies
    - Cycle mode (0): Use mean-reversion strategies

    Typical interpretation:
    - Value = 1: Market is trending, momentum strategies work better
    - Value = 0: Market is cycling, mean-reversion strategies work better
    - Mode changes: Market transitioning between trend and cycle
    - Stable mode: Current market condition is persistent

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ht_trendmode = HT_TRENDMODE(frame=frame, column_name='HT_TRENDMODE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ht_trendmode.periods[-1].HT_TRENDMODE)
        >>> print(ht_trendmode.is_trending())
        >>> print(ht_trendmode.is_cycling())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'HT_TRENDMODE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_TRENDMODE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'HT_TRENDMODE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_TRENDMODE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_TRENDMODE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 63 periods for HT_TRENDMODE calculation
        if period_index is None or len(self.frame.periods) < 63:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate HT_TRENDMODE using TA-Lib
        ht_trendmode_values = talib.HT_TRENDMODE(close_prices)

        # The last value is the HT_TRENDMODE for our period
        ht_trendmode_value = ht_trendmode_values[-1]

        if not np.isnan(ht_trendmode_value):
            # Convert to integer (0 or 1)
            setattr(period, self.column_name, int(ht_trendmode_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export HT_TRENDMODE values as numpy array.

        Returns:
            NumPy array with HT_TRENDMODE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[int]:
        """
        Get the latest HT_TRENDMODE value.

        Returns:
            1 for trend mode, 0 for cycle mode, or None if not available
        """
        if self.periods:
            value = getattr(self.periods[-1], self.column_name, None)
            return int(value) if value is not None else None
        return None

    def is_trending(self) -> bool:
        """
        Check if market is in trend mode.

        Returns:
            True if in trend mode (value = 1)
        """
        latest = self.get_latest()
        return latest is not None and latest == 1

    def is_cycling(self) -> bool:
        """
        Check if market is in cycle mode.

        Returns:
            True if in cycle mode (value = 0)
        """
        latest = self.get_latest()
        return latest is not None and latest == 0

    def mode_changed(self) -> Optional[str]:
        """
        Detect mode change.

        Returns:
            'to_trend' if changed to trend mode,
            'to_cycle' if changed to cycle mode,
            None if no change or not enough data
        """
        if len(self.periods) < 2:
            return None

        prev_mode = getattr(self.periods[-2], self.column_name, None)
        curr_mode = getattr(self.periods[-1], self.column_name, None)

        if prev_mode is not None and curr_mode is not None:
            if prev_mode == 0 and curr_mode == 1:
                return 'to_trend'
            elif prev_mode == 1 and curr_mode == 0:
                return 'to_cycle'

        return None

    def get_mode_stability(self, lookback: int = 10) -> Optional[float]:
        """
        Calculate mode stability (percentage of periods in same mode).

        Args:
            lookback: Number of periods to look back (default: 10)

        Returns:
            Stability percentage (0-100) or None
        """
        if len(self.periods) < lookback:
            return None

        current_mode = self.get_latest()
        if current_mode is None:
            return None

        same_mode_count = sum(
            1 for p in self.periods[-lookback:]
            if getattr(p, self.column_name, None) == current_mode
        )

        return (same_mode_count / lookback) * 100

    def get_trend_duration(self) -> int:
        """
        Get number of consecutive periods in trend mode.

        Returns:
            Number of consecutive trend periods (0 if in cycle mode)
        """
        if not self.is_trending():
            return 0

        count = 0
        for period in reversed(self.periods):
            mode = getattr(period, self.column_name, None)
            if mode == 1:
                count += 1
            else:
                break

        return count

    def get_cycle_duration(self) -> int:
        """
        Get number of consecutive periods in cycle mode.

        Returns:
            Number of consecutive cycle periods (0 if in trend mode)
        """
        if not self.is_cycling():
            return 0

        count = 0
        for period in reversed(self.periods):
            mode = getattr(period, self.column_name, None)
            if mode == 0:
                count += 1
            else:
                break

        return count

    def get_mode_string(self) -> Optional[str]:
        """
        Get human-readable mode description.

        Returns:
            'TREND' or 'CYCLE' or None
        """
        latest = self.get_latest()
        if latest is None:
            return None
        return 'TREND' if latest == 1 else 'CYCLE'
