"""LINEARREG_ANGLE - Linear Regression Angle."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class LINEARREG_ANGLE(BaseIndicator):
    """
    LINEARREG_ANGLE - Linear Regression Angle in degrees.

    Calculates the angle of the linear regression line relative to the horizontal.
    Positive angles indicate uptrends, negative angles indicate downtrends.
    Steeper angles indicate stronger trends.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> angle = LINEARREG_ANGLE(frame=frame, length=14, column_name='REG_ANGLE')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check for steep uptrend
        >>> if angle.is_steep_uptrend(threshold=45):
        ...     print("Strong uptrend detected (angle > 45°)")

    Example (Static utility):
        >>> prices = np.array([100, 102, 104, 106, 108, 110, 112])
        >>> angles = LINEARREG_ANGLE.compute(prices, length=5)
        >>> print(f"Regression angle: {angles[-1]:.2f}°")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 14,
        column_name: str = 'LINEARREG_ANGLE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize LINEARREG_ANGLE indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for angle calculation (default: 14)
            column_name: Name for the indicator column (default: 'LINEARREG_ANGLE')
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
        Calculate LINEARREG_ANGLE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with LINEARREG_ANGLE value
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

        # Calculate LINEARREG_ANGLE using TA-Lib
        angle_values = talib.LINEARREG_ANGLE(close_prices, timeperiod=self.length)

        # The last value is the angle for our period
        angle_value = angle_values[-1]

        if not np.isnan(angle_value):
            setattr(period, self.column_name, float(angle_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 14
    ) -> np.ndarray:
        """
        Compute LINEARREG_ANGLE values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for angle calculation (default: 14)

        Returns:
            NumPy array with angle values in degrees
        """
        return talib.LINEARREG_ANGLE(prices, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export LINEARREG_ANGLE values as numpy array.

        Returns:
            NumPy array with angle values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest LINEARREG_ANGLE value.

        Returns:
            Latest angle value in degrees or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_steep_uptrend(self, threshold: float = 45) -> bool:
        """
        Check if regression line shows steep uptrend.

        Args:
            threshold: Angle threshold in degrees (default: 45)

        Returns:
            True if angle > threshold
        """
        angle = self.get_latest()
        return angle is not None and angle > threshold

    def is_steep_downtrend(self, threshold: float = -45) -> bool:
        """
        Check if regression line shows steep downtrend.

        Args:
            threshold: Angle threshold in degrees (default: -45)

        Returns:
            True if angle < threshold
        """
        angle = self.get_latest()
        return angle is not None and angle < threshold

    def is_flat(self, threshold: float = 10) -> bool:
        """
        Check if regression line is relatively flat (low angle).

        Args:
            threshold: Maximum absolute angle for "flat" (default: 10)

        Returns:
            True if abs(angle) < threshold
        """
        angle = self.get_latest()
        return angle is not None and abs(angle) < threshold
