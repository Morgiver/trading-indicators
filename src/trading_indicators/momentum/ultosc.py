"""ULTOSC (Ultimate Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ULTOSC(BaseIndicator):
    """
    ULTOSC (Ultimate Oscillator) indicator.

    Ultimate Oscillator uses weighted sums of three oscillators,
    each calculated over different time periods.

    Typical interpretation:
    - ULTOSC > 70: Overbought
    - ULTOSC < 30: Oversold
    - ULTOSC crossing above 30: Bullish signal
    - ULTOSC crossing below 70: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ultosc = ULTOSC(frame=frame, period1=7, period2=14, period3=28)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(ultosc.periods[-1].ULTOSC)
        >>> print(ultosc.is_overbought())
    """

    def __init__(
        self,
        frame: 'Frame',
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        column_name: str = 'ULTOSC',
        max_periods: Optional[int] = None
    ):
        """
        Initialize ULTOSC indicator.

        Args:
            frame: Frame to bind to
            period1: First time period (default: 7)
            period2: Second time period (default: 14)
            period3: Third time period (default: 28)
            column_name: Name for the indicator column (default: 'ULTOSC')
            max_periods: Maximum periods to keep
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate ULTOSC value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least period3 + 1 for calculation
        required_periods = self.period3 + 1
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        ultosc_values = talib.ULTOSC(
            high_prices, low_prices, close_prices,
            timeperiod1=self.period1,
            timeperiod2=self.period2,
            timeperiod3=self.period3
        )
        ultosc_value = ultosc_values[-1]

        if not np.isnan(ultosc_value):
            setattr(period, self.column_name, float(ultosc_value))

    def to_numpy(self) -> np.ndarray:
        """Export ULTOSC values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """Export normalized ULTOSC values (0-100 → 0-1)."""
        return self.to_numpy() / 100.0

    def get_latest(self) -> Optional[float]:
        """Get the latest ULTOSC value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = 70.0) -> bool:
        """Check if ULTOSC indicates overbought condition."""
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = 30.0) -> bool:
        """Check if ULTOSC indicates oversold condition."""
        latest = self.get_latest()
        return latest is not None and latest < threshold
