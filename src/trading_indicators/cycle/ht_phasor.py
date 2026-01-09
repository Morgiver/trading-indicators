"""HT_PHASOR (Hilbert Transform - Phasor Components) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class HT_PHASOR(BaseIndicator):
    """
    HT_PHASOR (Hilbert Transform - Phasor Components) indicator.

    Uses Hilbert Transform to compute the in-phase and quadrature components
    of the dominant cycle. These components represent the cycle's position
    in 2D space (complex plane).

    Components:
    - InPhase: In-phase component (real part)
    - Quadrature: Quadrature component (imaginary part)

    Typical interpretation:
    - InPhase crossing zero: Cycle turning point
    - Quadrature leading InPhase by 90°
    - Phase angle = atan2(Quadrature, InPhase)
    - Magnitude = sqrt(InPhase² + Quadrature²)
    - Used for cycle analysis and prediction

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ht_phasor = HT_PHASOR(frame=frame)
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ht_phasor.periods[-1].HT_INPHASE)
        >>> print(ht_phasor.periods[-1].HT_QUADRATURE)
        >>> print(ht_phasor.get_magnitude())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_PHASOR indicator.

        Args:
            frame: Frame to bind to
            column_names: Names for [inphase, quadrature] (default: ['HT_INPHASE', 'HT_QUADRATURE'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_names = column_names or ['HT_INPHASE', 'HT_QUADRATURE']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [inphase, quadrature]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_PHASOR values for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_PHASOR values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 63 periods for HT_PHASOR calculation
        if period_index is None or len(self.frame.periods) < 63:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate HT_PHASOR using TA-Lib
        inphase, quadrature = talib.HT_PHASOR(close_prices)

        # The last values are for our period
        if not np.isnan(inphase[-1]):
            setattr(period, self.column_names[0], float(inphase[-1]))

        if not np.isnan(quadrature[-1]):
            setattr(period, self.column_names[1], float(quadrature[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export HT_PHASOR values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'HT_INPHASE', 'HT_QUADRATURE'
            (or custom column names) mapping to numpy arrays
        """
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def get_latest(self) -> Optional[Dict[str, float]]:
        """
        Get the latest HT_PHASOR values.

        Returns:
            Dictionary with latest InPhase and Quadrature values or None
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def get_magnitude(self) -> Optional[float]:
        """
        Calculate phasor magnitude (cycle strength).

        Formula: sqrt(InPhase² + Quadrature²)

        Returns:
            Magnitude of the phasor or None
        """
        latest = self.get_latest()
        if latest and all(name in latest for name in self.column_names):
            inphase = latest[self.column_names[0]]
            quadrature = latest[self.column_names[1]]
            return np.sqrt(inphase**2 + quadrature**2)
        return None

    def get_phase_angle(self) -> Optional[float]:
        """
        Calculate phase angle in degrees.

        Formula: atan2(Quadrature, InPhase) * 180 / π

        Returns:
            Phase angle in degrees (-180 to 180) or None
        """
        latest = self.get_latest()
        if latest and all(name in latest for name in self.column_names):
            inphase = latest[self.column_names[0]]
            quadrature = latest[self.column_names[1]]
            angle_rad = np.arctan2(quadrature, inphase)
            return np.degrees(angle_rad)
        return None

    def is_inphase_positive(self) -> bool:
        """
        Check if InPhase component is positive.

        Returns:
            True if InPhase > 0
        """
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] > 0
        return False

    def is_inphase_crossing_zero(self) -> Optional[str]:
        """
        Detect InPhase zero crossing.

        Returns:
            'up' if crossing from negative to positive,
            'down' if crossing from positive to negative,
            None if no crossing or not enough data
        """
        if len(self.periods) < 2:
            return None

        prev_inphase = getattr(self.periods[-2], self.column_names[0], None)
        curr_inphase = getattr(self.periods[-1], self.column_names[0], None)

        if prev_inphase is not None and curr_inphase is not None:
            if prev_inphase <= 0 and curr_inphase > 0:
                return 'up'
            elif prev_inphase >= 0 and curr_inphase < 0:
                return 'down'

        return None

    def get_cycle_position(self) -> Optional[str]:
        """
        Determine position in cycle based on phasor components.

        Returns:
            'bottom', 'rising', 'top', 'falling', or None
        """
        angle = self.get_phase_angle()
        if angle is None:
            return None

        # Convert to 0-360 range
        if angle < 0:
            angle += 360

        if 315 <= angle or angle < 45:
            return 'bottom'
        elif 45 <= angle < 135:
            return 'rising'
        elif 135 <= angle < 225:
            return 'top'
        else:
            return 'falling'
