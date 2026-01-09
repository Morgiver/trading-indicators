"""CMO (Chande Momentum Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class CMO(BaseIndicator):
    """
    CMO (Chande Momentum Oscillator) indicator.

    CMO is similar to RSI but uses the sum of gains and losses instead
    of averages. It ranges from -100 to +100.

    Typical interpretation:
    - CMO > +50: Overbought
    - CMO < -50: Oversold
    - CMO crossing above 0: Bullish signal
    - CMO crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> cmo = CMO(frame=frame, length=14, column_name='CMO_14')
        >>>
        >>> # Feed candles - CMO automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(cmo.periods[-1].CMO_14)
        >>> print(cmo.is_overbought())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'CMO',
        max_periods: Optional[int] = None
    ):
        """
        Initialize CMO indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for CMO calculation (default: 14)
            column_name: Name for the indicator column (default: 'CMO')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate CMO value for a specific period.

        Args:
            period: IndicatorPeriod to populate with CMO value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length + 1' periods for CMO calculation
        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate CMO using TA-Lib
        cmo_values = talib.CMO(close_prices, timeperiod=self.length)

        # The last value is the CMO for our period
        cmo_value = cmo_values[-1]

        if not np.isnan(cmo_value):
            setattr(period, self.column_name, float(cmo_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export CMO values as numpy array.

        Returns:
            NumPy array with CMO values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized CMO values for ML (-100 to +100 → 0-1).

        Returns:
            NumPy array with normalized CMO values [0, 1]
        """
        values = self.to_numpy()
        return (values + 100.0) / 200.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest CMO value.

        Returns:
            Latest CMO value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = 50.0) -> bool:
        """
        Check if CMO indicates overbought condition.

        Args:
            threshold: Overbought threshold (default: 50.0)

        Returns:
            True if CMO is above threshold
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = -50.0) -> bool:
        """
        Check if CMO indicates oversold condition.

        Args:
            threshold: Oversold threshold (default: -50.0)

        Returns:
            True if CMO is below threshold
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold

    def is_bullish(self) -> bool:
        """
        Check if CMO indicates bullish condition (CMO > 0).

        Returns:
            True if CMO is positive
        """
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """
        Check if CMO indicates bearish condition (CMO < 0).

        Returns:
            True if CMO is negative
        """
        latest = self.get_latest()
        return latest is not None and latest < 0
