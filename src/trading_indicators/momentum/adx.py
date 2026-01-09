"""ADX (Average Directional Movement Index) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ADX(BaseIndicator):
    """
    ADX (Average Directional Movement Index) indicator.

    ADX measures the strength of a trend regardless of direction.
    Values range from 0 to 100.

    Typical interpretation:
    - ADX < 20: Weak trend (ranging market)
    - ADX 20-40: Moderate trend
    - ADX > 40: Strong trend
    - ADX > 50: Very strong trend

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> adx = ADX(frame=frame, length=14, column_name='ADX_14')
        >>>
        >>> # Feed candles - ADX automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(adx.periods[-1].ADX_14)
        >>> print(adx.is_trending())
        >>> print(adx.is_strong_trend())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'ADX',
        max_periods: Optional[int] = None
    ):
        """
        Initialize ADX indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for ADX calculation (default: 14)
            column_name: Name for the indicator column (default: 'ADX')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate ADX value for a specific period.

        Args:
            period: IndicatorPeriod to populate with ADX value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'length * 2' periods for ADX calculation
        required_periods = self.length * 2
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        # Extract OHLC prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate ADX using TA-Lib
        adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.length)

        # The last value is the ADX for our period
        adx_value = adx_values[-1]

        if not np.isnan(adx_value):
            setattr(period, self.column_name, float(adx_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export ADX values as numpy array.

        Returns:
            NumPy array with ADX values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized ADX values for ML (0-100 → 0-1).

        Returns:
            NumPy array with normalized ADX values [0, 1]
        """
        values = self.to_numpy()
        return values / 100.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest ADX value.

        Returns:
            Latest ADX value or None if not available
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
            True if ADX is above threshold (market is trending)
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_strong_trend(self, threshold: float = 40.0) -> bool:
        """
        Check if market is in strong trend.

        Args:
            threshold: Strong trend threshold (default: 40.0)

        Returns:
            True if ADX is above threshold (strong trend)
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold
