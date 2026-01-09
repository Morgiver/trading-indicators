"""AROON indicator."""

from typing import Optional, Dict, List
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class AROON(BaseIndicator):
    """
    AROON indicator.

    Aroon identifies trend changes and strength by measuring time
    since the highest and lowest prices over a period.

    Components:
    - Aroon Up: Time since highest high
    - Aroon Down: Time since lowest low

    Typical interpretation:
    - Aroon Up > 70 and Aroon Down < 30: Strong uptrend
    - Aroon Down > 70 and Aroon Up < 30: Strong downtrend
    - Both near 50: Consolidation
    - Aroon Up crosses above Aroon Down: Bullish signal
    - Aroon Down crosses above Aroon Up: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> aroon = AROON(frame=frame, length=14)
        >>>
        >>> # Feed candles - AROON automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(aroon.periods[-1].AROON_UP)
        >>> print(aroon.periods[-1].AROON_DOWN)
        >>> print(aroon.is_uptrend())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize AROON indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for AROON calculation (default: 14)
            column_names: Names for [aroon_down, aroon_up] (default: ['AROON_DOWN', 'AROON_UP'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_names = column_names or ['AROON_DOWN', 'AROON_UP']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [aroon_down, aroon_up]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate AROON values for a specific period.

        Args:
            period: IndicatorPeriod to populate with AROON values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length + 1' periods for AROON calculation
        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        # Extract high and low prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])

        # Calculate AROON using TA-Lib
        aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=self.length)

        # The last values are for our period
        if not np.isnan(aroon_down[-1]):
            setattr(period, self.column_names[0], float(aroon_down[-1]))

        if not np.isnan(aroon_up[-1]):
            setattr(period, self.column_names[1], float(aroon_up[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export AROON values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'AROON_DOWN', 'AROON_UP'
            (or custom column names) mapping to numpy arrays
        """
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def to_normalize(self) -> Dict[str, np.ndarray]:
        """
        Export normalized AROON values for ML (0-100 → 0-1).

        Returns:
            Dictionary with normalized arrays [0, 1]
        """
        arrays = self.to_numpy()
        return {key: arr / 100.0 for key, arr in arrays.items()}

    def get_latest(self) -> Optional[Dict[str, float]]:
        """
        Get the latest AROON values.

        Returns:
            Dictionary with latest AROON values or None if not available
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_uptrend(self, up_threshold: float = 70.0, down_threshold: float = 30.0) -> bool:
        """
        Check if indicator shows strong uptrend.

        Args:
            up_threshold: Aroon Up threshold (default: 70.0)
            down_threshold: Aroon Down threshold (default: 30.0)

        Returns:
            True if Aroon Up > up_threshold and Aroon Down < down_threshold
        """
        latest = self.get_latest()
        if latest and self.column_names[0] in latest and self.column_names[1] in latest:
            aroon_down = latest[self.column_names[0]]
            aroon_up = latest[self.column_names[1]]
            return aroon_up > up_threshold and aroon_down < down_threshold
        return False

    def is_downtrend(self, down_threshold: float = 70.0, up_threshold: float = 30.0) -> bool:
        """
        Check if indicator shows strong downtrend.

        Args:
            down_threshold: Aroon Down threshold (default: 70.0)
            up_threshold: Aroon Up threshold (default: 30.0)

        Returns:
            True if Aroon Down > down_threshold and Aroon Up < up_threshold
        """
        latest = self.get_latest()
        if latest and self.column_names[0] in latest and self.column_names[1] in latest:
            aroon_down = latest[self.column_names[0]]
            aroon_up = latest[self.column_names[1]]
            return aroon_down > down_threshold and aroon_up < up_threshold
        return False
