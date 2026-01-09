"""VAR - Variance."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class VAR(BaseIndicator):
    """
    VAR - Variance (statistical measure of dispersion).

    Calculates the variance of price values, which is the square of standard deviation.
    Variance measures how far prices spread from their mean value.
    Higher variance indicates higher volatility and price dispersion.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> var = VAR(frame=frame, length=20, nbdev=1, column_name='VAR')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check for high variance
        >>> if var.is_high_variance(threshold=4.0):
        ...     print("High variance/volatility detected")

    Example (Static utility):
        >>> prices = np.array([100, 102, 98, 105, 97, 103, 99])
        >>> var_values = VAR.compute(prices, length=5, nbdev=1)
        >>> print(f"Current variance: {var_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 20,
        nbdev: float = 1,
        column_name: str = 'VAR',
        max_periods: Optional[int] = None
    ):
        """
        Initialize VAR indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for variance calculation (default: 20)
            nbdev: Number of standard deviations (default: 1)
            column_name: Name for the indicator column (default: 'VAR')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.nbdev = nbdev
        self.column_name = column_name

        if frame:
            super().__init__(frame, max_periods)
        else:
            self.frame = None
            self.periods = []

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate VAR value for a specific period.

        Args:
            period: IndicatorPeriod to populate with VAR value
        """
        if len(self.frame.periods) < self.length:
            return

        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate VAR using TA-Lib
        var_values = talib.VAR(close_prices, timeperiod=self.length, nbdev=self.nbdev)

        # The last value is the variance for our period
        var_value = var_values[-1]

        if not np.isnan(var_value):
            setattr(period, self.column_name, float(var_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 20,
        nbdev: float = 1
    ) -> np.ndarray:
        """
        Compute VAR values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for variance calculation (default: 20)
            nbdev: Number of standard deviations (default: 1)

        Returns:
            NumPy array with variance values
        """
        return talib.VAR(prices, timeperiod=length, nbdev=nbdev)

    def to_numpy(self) -> np.ndarray:
        """
        Export VAR values as numpy array.

        Returns:
            NumPy array with VAR values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest VAR value.

        Returns:
            Latest variance value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_high_variance(self, threshold: float = 4.0) -> bool:
        """
        Check if variance is high (var > threshold).

        Args:
            threshold: Variance threshold (default: 4.0)

        Returns:
            True if variance > threshold
        """
        var = self.get_latest()
        return var is not None and var > threshold

    def is_low_variance(self, threshold: float = 1.0) -> bool:
        """
        Check if variance is low (var < threshold).

        Args:
            threshold: Variance threshold (default: 1.0)

        Returns:
            True if variance < threshold
        """
        var = self.get_latest()
        return var is not None and var < threshold
