"""HT_DCPHASE (Hilbert Transform - Dominant Cycle Phase) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class HT_DCPHASE(BaseIndicator):
    """
    HT_DCPHASE (Hilbert Transform - Dominant Cycle Phase) indicator.

    Uses Hilbert Transform to identify the current phase angle of the dominant cycle.
    The phase ranges from 0 to 360 degrees and helps identify where we are in
    the current market cycle.

    Typical interpretation:
    - 0-90°: Early uptrend phase
    - 90-180°: Late uptrend phase, potential top
    - 180-270°: Early downtrend phase
    - 270-360°: Late downtrend phase, potential bottom
    - Phase changing rapidly: Strong momentum
    - Phase stable: Consolidation

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ht_dcphase = HT_DCPHASE(frame=frame, column_name='HT_DCPHASE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ht_dcphase.periods[-1].HT_DCPHASE)
        >>> print(ht_dcphase.get_phase_quadrant())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'HT_DCPHASE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_DCPHASE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'HT_DCPHASE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_DCPHASE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_DCPHASE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 63 periods for HT_DCPHASE calculation
        if period_index is None or len(self.frame.periods) < 63:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate HT_DCPHASE using TA-Lib
        ht_dcphase_values = talib.HT_DCPHASE(close_prices)

        # The last value is the HT_DCPHASE for our period
        ht_dcphase_value = ht_dcphase_values[-1]

        if not np.isnan(ht_dcphase_value):
            setattr(period, self.column_name, float(ht_dcphase_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export HT_DCPHASE values as numpy array.

        Returns:
            NumPy array with HT_DCPHASE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest HT_DCPHASE value.

        Returns:
            Latest HT_DCPHASE value (0-360 degrees) or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def get_phase_quadrant(self) -> Optional[str]:
        """
        Get the current phase quadrant.

        Returns:
            'Q1' (0-90°), 'Q2' (90-180°), 'Q3' (180-270°), 'Q4' (270-360°), or None
        """
        latest = self.get_latest()
        if latest is None:
            return None

        if 0 <= latest < 90:
            return 'Q1'  # Early uptrend
        elif 90 <= latest < 180:
            return 'Q2'  # Late uptrend
        elif 180 <= latest < 270:
            return 'Q3'  # Early downtrend
        else:
            return 'Q4'  # Late downtrend

    def is_early_uptrend(self) -> bool:
        """
        Check if phase indicates early uptrend (0-90°).

        Returns:
            True if in Q1 (early uptrend phase)
        """
        return self.get_phase_quadrant() == 'Q1'

    def is_late_uptrend(self) -> bool:
        """
        Check if phase indicates late uptrend (90-180°).

        Returns:
            True if in Q2 (late uptrend, potential top)
        """
        return self.get_phase_quadrant() == 'Q2'

    def is_early_downtrend(self) -> bool:
        """
        Check if phase indicates early downtrend (180-270°).

        Returns:
            True if in Q3 (early downtrend phase)
        """
        return self.get_phase_quadrant() == 'Q3'

    def is_late_downtrend(self) -> bool:
        """
        Check if phase indicates late downtrend (270-360°).

        Returns:
            True if in Q4 (late downtrend, potential bottom)
        """
        return self.get_phase_quadrant() == 'Q4'

    def get_phase_velocity(self, lookback: int = 3) -> Optional[float]:
        """
        Calculate phase change rate (velocity).

        Args:
            lookback: Number of periods to look back (default: 3)

        Returns:
            Phase change in degrees per period, or None
        """
        if len(self.periods) < lookback + 1:
            return None

        old_phase = getattr(self.periods[-lookback-1], self.column_name, None)
        curr_phase = getattr(self.periods[-1], self.column_name, None)

        if old_phase is not None and curr_phase is not None:
            # Handle phase wrap-around (360° to 0°)
            phase_diff = curr_phase - old_phase
            if phase_diff < -180:
                phase_diff += 360
            elif phase_diff > 180:
                phase_diff -= 360

            return phase_diff / lookback

        return None

    def is_phase_accelerating(self) -> bool:
        """
        Check if phase is changing rapidly (high velocity).

        Returns:
            True if phase velocity is high (> 30° per period)
        """
        velocity = self.get_phase_velocity()
        return velocity is not None and abs(velocity) > 30
