"""ADXR (Average Directional Movement Index Rating) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ADXR(BaseIndicator):
    """
    ADXR (Average Directional Movement Index Rating) indicator.

    ADXR is the average of the current ADX and the ADX from n periods ago.
    It provides a smoother trend strength measurement than ADX alone.

    Typical interpretation:
    - ADXR < 20: Weak trend
    - ADXR 20-40: Moderate trend
    - ADXR > 40: Strong trend

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> adxr = ADXR(frame=frame, length=14, column_name='ADXR_14')
        >>>
        >>> # Feed candles - ADXR automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(adxr.periods[-1].ADXR_14)
        >>> print(adxr.is_trending())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'ADXR',
        max_periods: Optional[int] = None
    ):
        """
        Initialize ADXR indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for ADXR calculation (default: 14)
            column_name: Name for the indicator column (default: 'ADXR')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate ADXR value for a specific period.

        Args:
            period: IndicatorPeriod to populate with ADXR value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length * 3' periods for ADXR calculation
        required_periods = self.length * 3
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        # Extract OHLC prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate ADXR using TA-Lib
        adxr_values = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=self.length)

        # The last value is the ADXR for our period
        adxr_value = adxr_values[-1]

        if not np.isnan(adxr_value):
            setattr(period, self.column_name, float(adxr_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export ADXR values as numpy array.

        Returns:
            NumPy array with ADXR values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized ADXR values for ML (0-100 → 0-1).

        Returns:
            NumPy array with normalized ADXR values [0, 1]
        """
        values = self.to_numpy()
        return values / 100.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest ADXR value.

        Returns:
            Latest ADXR value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_trending(self, threshold: float = 20.0) -> bool:
        """
        Check if market is trending.

        Args:
            threshold: Trend threshold (default: 20.0)

        Returns:
            True if ADXR is above threshold (market is trending)
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_strong_trend(self, threshold: float = 40.0) -> bool:
        """
        Check if market is in strong trend.

        Args:
            threshold: Strong trend threshold (default: 40.0)

        Returns:
            True if ADXR is above threshold (strong trend)
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold
