"""AROONOSC (Aroon Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class AROONOSC(BaseIndicator):
    """
    AROONOSC (Aroon Oscillator) indicator.

    Aroon Oscillator is the difference between Aroon Up and Aroon Down.
    It ranges from -100 to +100.

    Typical interpretation:
    - AROONOSC > 0: Uptrend (Aroon Up > Aroon Down)
    - AROONOSC < 0: Downtrend (Aroon Down > Aroon Up)
    - AROONOSC near +100: Very strong uptrend
    - AROONOSC near -100: Very strong downtrend
    - AROONOSC near 0: Consolidation

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> aroonosc = AROONOSC(frame=frame, length=14, column_name='AROONOSC')
        >>>
        >>> # Feed candles - AROONOSC automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(aroonosc.periods[-1].AROONOSC)
        >>> print(aroonosc.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'AROONOSC',
        max_periods: Optional[int] = None
    ):
        """
        Initialize AROONOSC indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for AROONOSC calculation (default: 14)
            column_name: Name for the indicator column (default: 'AROONOSC')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate AROONOSC value for a specific period.

        Args:
            period: IndicatorPeriod to populate with AROONOSC value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length + 1' periods for AROONOSC calculation
        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        # Extract high and low prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])

        # Calculate AROONOSC using TA-Lib
        aroonosc_values = talib.AROONOSC(high_prices, low_prices, timeperiod=self.length)

        # The last value is the AROONOSC for our period
        aroonosc_value = aroonosc_values[-1]

        if not np.isnan(aroonosc_value):
            setattr(period, self.column_name, float(aroonosc_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export AROONOSC values as numpy array.

        Returns:
            NumPy array with AROONOSC values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized AROONOSC values for ML (-100 to +100 → 0-1).

        Returns:
            NumPy array with normalized AROONOSC values [0, 1]
        """
        values = self.to_numpy()
        return (values + 100.0) / 200.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest AROONOSC value.

        Returns:
            Latest AROONOSC value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """
        Check if AROONOSC indicates bullish condition (AROONOSC > 0).

        Returns:
            True if AROONOSC is positive
        """
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """
        Check if AROONOSC indicates bearish condition (AROONOSC < 0).

        Returns:
            True if AROONOSC is negative
        """
        latest = self.get_latest()
        return latest is not None and latest < 0

    def is_strong_uptrend(self, threshold: float = 50.0) -> bool:
        """
        Check if AROONOSC indicates strong uptrend.

        Args:
            threshold: Strong uptrend threshold (default: 50.0)

        Returns:
            True if AROONOSC is above threshold
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_strong_downtrend(self, threshold: float = -50.0) -> bool:
        """
        Check if AROONOSC indicates strong downtrend.

        Args:
            threshold: Strong downtrend threshold (default: -50.0)

        Returns:
            True if AROONOSC is below threshold
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold
