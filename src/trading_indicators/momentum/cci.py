"""CCI (Commodity Channel Index) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class CCI(BaseIndicator):
    """
    CCI (Commodity Channel Index) indicator.

    CCI measures the deviation of price from its statistical mean.
    It's useful for identifying cyclical trends and overbought/oversold conditions.

    Typical interpretation:
    - CCI > +100: Overbought, strong uptrend
    - CCI < -100: Oversold, strong downtrend
    - CCI crossing above +100: Buy signal
    - CCI crossing below -100: Sell signal
    - CCI between -100 and +100: Normal trading range

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> cci = CCI(frame=frame, length=14, column_name='CCI_14')
        >>>
        >>> # Feed candles - CCI automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(cci.periods[-1].CCI_14)
        >>> print(cci.is_overbought())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'CCI',
        max_periods: Optional[int] = None
    ):
        """
        Initialize CCI indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for CCI calculation (default: 14)
            column_name: Name for the indicator column (default: 'CCI')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate CCI value for a specific period.

        Args:
            period: IndicatorPeriod to populate with CCI value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length' periods for CCI calculation
        if period_index is None or len(self.frame.periods) < self.length:
            return

        # Extract OHLC prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate CCI using TA-Lib
        cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=self.length)

        # The last value is the CCI for our period
        cci_value = cci_values[-1]

        if not np.isnan(cci_value):
            setattr(period, self.column_name, float(cci_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export CCI values as numpy array.

        Returns:
            NumPy array with CCI values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest CCI value.

        Returns:
            Latest CCI value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = 100.0) -> bool:
        """
        Check if CCI indicates overbought condition.

        Args:
            threshold: Overbought threshold (default: 100.0)

        Returns:
            True if CCI is above threshold
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = -100.0) -> bool:
        """
        Check if CCI indicates oversold condition.

        Args:
            threshold: Oversold threshold (default: -100.0)

        Returns:
            True if CCI is below threshold
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold
