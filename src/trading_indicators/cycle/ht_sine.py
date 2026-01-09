"""HT_SINE (Hilbert Transform - SineWave) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class HT_SINE(BaseIndicator):
    """
    HT_SINE (Hilbert Transform - SineWave) indicator.

    Uses Hilbert Transform to generate a sine wave and lead sine wave
    that represent the dominant cycle. The lead sine is ahead by 45°,
    providing early signals.

    Components:
    - Sine: Sine wave of dominant cycle
    - LeadSine: Lead sine wave (45° ahead)

    Typical interpretation:
    - Sine crossing above LeadSine: Potential buy signal
    - Sine crossing below LeadSine: Potential sell signal
    - Both near +1: Overbought (cycle peak)
    - Both near -1: Oversold (cycle trough)
    - Crossovers indicate cycle turning points

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ht_sine = HT_SINE(frame=frame)
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ht_sine.periods[-1].HT_SINE)
        >>> print(ht_sine.periods[-1].HT_LEADSINE)
        >>> print(ht_sine.is_bullish_crossover())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_SINE indicator.

        Args:
            frame: Frame to bind to
            column_names: Names for [sine, leadsine] (default: ['HT_SINE', 'HT_LEADSINE'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_names = column_names or ['HT_SINE', 'HT_LEADSINE']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [sine, leadsine]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_SINE values for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_SINE values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 63 periods for HT_SINE calculation
        if period_index is None or len(self.frame.periods) < 63:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate HT_SINE using TA-Lib
        sine, leadsine = talib.HT_SINE(close_prices)

        # The last values are for our period
        if not np.isnan(sine[-1]):
            setattr(period, self.column_names[0], float(sine[-1]))

        if not np.isnan(leadsine[-1]):
            setattr(period, self.column_names[1], float(leadsine[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export HT_SINE values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'HT_SINE', 'HT_LEADSINE'
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
        Get the latest HT_SINE values.

        Returns:
            Dictionary with latest Sine and LeadSine values or None
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_bullish_crossover(self) -> bool:
        """
        Check if Sine crossed above LeadSine (bullish signal).

        Returns:
            True if bullish crossover detected
        """
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_sine = getattr(prev, self.column_names[0], None)
        prev_lead = getattr(prev, self.column_names[1], None)
        curr_sine = getattr(curr, self.column_names[0], None)
        curr_lead = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_sine, prev_lead, curr_sine, curr_lead]):
            return prev_sine <= prev_lead and curr_sine > curr_lead

        return False

    def is_bearish_crossover(self) -> bool:
        """
        Check if Sine crossed below LeadSine (bearish signal).

        Returns:
            True if bearish crossover detected
        """
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_sine = getattr(prev, self.column_names[0], None)
        prev_lead = getattr(prev, self.column_names[1], None)
        curr_sine = getattr(curr, self.column_names[0], None)
        curr_lead = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_sine, prev_lead, curr_sine, curr_lead]):
            return prev_sine >= prev_lead and curr_sine < curr_lead

        return False

    def is_overbought(self, threshold: float = 0.7) -> bool:
        """
        Check if cycle indicates overbought condition.

        Args:
            threshold: Overbought threshold (default: 0.7)

        Returns:
            True if both Sine and LeadSine are above threshold
        """
        latest = self.get_latest()
        if latest and all(name in latest for name in self.column_names):
            sine = latest[self.column_names[0]]
            leadsine = latest[self.column_names[1]]
            return sine > threshold and leadsine > threshold
        return False

    def is_oversold(self, threshold: float = -0.7) -> bool:
        """
        Check if cycle indicates oversold condition.

        Args:
            threshold: Oversold threshold (default: -0.7)

        Returns:
            True if both Sine and LeadSine are below threshold
        """
        latest = self.get_latest()
        if latest and all(name in latest for name in self.column_names):
            sine = latest[self.column_names[0]]
            leadsine = latest[self.column_names[1]]
            return sine < threshold and leadsine < threshold
        return False

    def get_cycle_strength(self) -> Optional[float]:
        """
        Calculate cycle strength based on sine amplitude.

        Returns:
            Average absolute value of sine waves (0-1) or None
        """
        latest = self.get_latest()
        if latest and all(name in latest for name in self.column_names):
            sine = abs(latest[self.column_names[0]])
            leadsine = abs(latest[self.column_names[1]])
            return (sine + leadsine) / 2
        return None

    def is_at_cycle_peak(self, threshold: float = 0.9) -> bool:
        """
        Check if at cycle peak.

        Args:
            threshold: Peak threshold (default: 0.9)

        Returns:
            True if near cycle peak
        """
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] > threshold
        return False

    def is_at_cycle_trough(self, threshold: float = -0.9) -> bool:
        """
        Check if at cycle trough.

        Args:
            threshold: Trough threshold (default: -0.9)

        Returns:
            True if near cycle trough
        """
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] < threshold
        return False
