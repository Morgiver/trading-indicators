"""LINEARREG_SLOPE - Linear Regression Slope."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class LINEARREG_SLOPE(BaseIndicator):
    """
    LINEARREG_SLOPE - Slope of the linear regression line.

    Calculates the slope (rate of change) of the linear regression line.
    Positive slope indicates upward trend, negative indicates downward trend.
    The magnitude represents the steepness of the trend.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> slope = LINEARREG_SLOPE(frame=frame, length=14, column_name='REG_SLOPE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check if trend is accelerating
        >>> if slope.is_accelerating():
        ...     print("Trend is accelerating")

    Example (Static utility):
        >>> prices = np.array([100, 102, 104, 106, 108, 110, 112])
        >>> slopes = LINEARREG_SLOPE.compute(prices, length=5)
        >>> print(f"Regression slope: {slopes[-1]:.4f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 14,
        column_name: str = 'LINEARREG_SLOPE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize LINEARREG_SLOPE indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for slope calculation (default: 14)
            column_name: Name for the indicator column (default: 'LINEARREG_SLOPE')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name

        if frame:
            super().__init__(frame, max_periods)
        else:
            self.frame = None
            self.periods = []

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate LINEARREG_SLOPE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with LINEARREG_SLOPE value
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

        # Calculate LINEARREG_SLOPE using TA-Lib
        slope_values = talib.LINEARREG_SLOPE(close_prices, timeperiod=self.length)

        # The last value is the slope for our period
        slope_value = slope_values[-1]

        if not np.isnan(slope_value):
            setattr(period, self.column_name, float(slope_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 14
    ) -> np.ndarray:
        """
        Compute LINEARREG_SLOPE values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for slope calculation (default: 14)

        Returns:
            NumPy array with slope values
        """
        return talib.LINEARREG_SLOPE(prices, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export LINEARREG_SLOPE values as numpy array.

        Returns:
            NumPy array with slope values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest LINEARREG_SLOPE value.

        Returns:
            Latest slope value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_positive_slope(self) -> bool:
        """
        Check if regression line has positive slope (uptrend).

        Returns:
            True if slope > 0
        """
        slope = self.get_latest()
        return slope is not None and slope > 0

    def is_negative_slope(self) -> bool:
        """
        Check if regression line has negative slope (downtrend).

        Returns:
            True if slope < 0
        """
        slope = self.get_latest()
        return slope is not None and slope < 0

    def is_accelerating(self) -> bool:
        """
        Check if trend is accelerating (slope magnitude increasing).

        Returns:
            True if abs(current_slope) > abs(previous_slope)
        """
        if len(self.periods) < 2:
            return False

        current_slope = getattr(self.periods[-1], self.column_name, None)
        previous_slope = getattr(self.periods[-2], self.column_name, None)

        if current_slope is None or previous_slope is None:
            return False

        return abs(current_slope) > abs(previous_slope)
