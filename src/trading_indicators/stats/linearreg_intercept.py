"""LINEARREG_INTERCEPT - Linear Regression Intercept."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class LINEARREG_INTERCEPT(BaseIndicator):
    """
    LINEARREG_INTERCEPT - Y-intercept of the linear regression line.

    Calculates the y-intercept (constant term) of the linear regression line.
    This represents where the regression line would cross the y-axis if extended.
    Useful for understanding the baseline level of the trend.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> intercept = LINEARREG_INTERCEPT(frame=frame, length=14, column_name='REG_INTERCEPT')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Get latest intercept value
        >>> latest = intercept.get_latest()
        >>> print(f"Regression intercept: {latest:.2f}")

    Example (Static utility):
        >>> prices = np.array([100, 102, 104, 106, 108, 110, 112])
        >>> intercepts = LINEARREG_INTERCEPT.compute(prices, length=5)
        >>> print(f"Latest intercept: {intercepts[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 14,
        column_name: str = 'LINEARREG_INTERCEPT',
        max_periods: Optional[int] = None
    ):
        """
        Initialize LINEARREG_INTERCEPT indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for intercept calculation (default: 14)
            column_name: Name for the indicator column (default: 'LINEARREG_INTERCEPT')
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
        Calculate LINEARREG_INTERCEPT value for a specific period.

        Args:
            period: IndicatorPeriod to populate with LINEARREG_INTERCEPT value
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

        # Calculate LINEARREG_INTERCEPT using TA-Lib
        intercept_values = talib.LINEARREG_INTERCEPT(close_prices, timeperiod=self.length)

        # The last value is the intercept for our period
        intercept_value = intercept_values[-1]

        if not np.isnan(intercept_value):
            setattr(period, self.column_name, float(intercept_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 14
    ) -> np.ndarray:
        """
        Compute LINEARREG_INTERCEPT values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for intercept calculation (default: 14)

        Returns:
            NumPy array with intercept values
        """
        return talib.LINEARREG_INTERCEPT(prices, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export LINEARREG_INTERCEPT values as numpy array.

        Returns:
            NumPy array with intercept values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest LINEARREG_INTERCEPT value.

        Returns:
            Latest intercept value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
