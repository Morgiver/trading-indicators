"""RSI (Relative Strength Index) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. Values range from 0 to 100.

    Typical interpretation:
    - RSI > 70: Overbought
    - RSI < 30: Oversold

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> rsi = RSI(frame=frame, length=14, column_name='RSI_14')
        >>>
        >>> # Feed candles - RSI automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(rsi.periods[-1].RSI_14)
        >>> print(rsi.is_overbought())
        >>> print(rsi.is_oversold())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'RSI',
        max_periods: Optional[int] = None
    ):
        """
        Initialize RSI indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for RSI calculation (default: 14)
            column_name: Name for the indicator column (default: 'RSI')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate RSI value for a specific period.

        Args:
            period: IndicatorPeriod to populate with RSI value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length + 1' periods for RSI calculation
        if period_index is None or period_index < self.length:
            return

        # Extract close prices for calculation
        close_prices = np.array([
            p.close_price
            for p in self.frame.periods[period_index - self.length:period_index + 1]
        ])

        # Calculate RSI using TA-Lib
        rsi_values = talib.RSI(close_prices, timeperiod=self.length)

        # The last value is the RSI for our period
        rsi_value = rsi_values[-1]

        if not np.isnan(rsi_value):
            setattr(period, self.column_name, float(rsi_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export RSI values as numpy array.

        Returns:
            NumPy array with RSI values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized RSI values for ML (0-100 → 0-1).

        Returns:
            NumPy array with normalized RSI values [0, 1]
        """
        values = self.to_numpy()
        return values / 100.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest RSI value.

        Returns:
            Latest RSI value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = 70.0) -> bool:
        """
        Check if RSI indicates overbought condition.

        Args:
            threshold: Overbought threshold (default: 70.0)

        Returns:
            True if RSI is above threshold
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = 30.0) -> bool:
        """
        Check if RSI indicates oversold condition.

        Args:
            threshold: Oversold threshold (default: 30.0)

        Returns:
            True if RSI is below threshold
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold
