"""LINEARREG - Linear Regression."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class LINEARREG(BaseIndicator):
    """
    LINEARREG - Linear Regression line values.

    Calculates the linear regression line fitted to price data over a specified period.
    The output is the value of the regression line at each point in time.
    Useful for identifying trend direction and support/resistance levels.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> linearreg = LINEARREG(frame=frame, length=14, column_name='LINEARREG')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check if price is above regression line
        >>> if linearreg.is_above_regression():
        ...     print("Price above linear regression trend")

    Example (Static utility):
        >>> prices = np.array([100, 102, 101, 103, 105, 104, 106])
        >>> regression_values = LINEARREG.compute(prices, length=5)
        >>> print(f"Latest regression value: {regression_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 14,
        column_name: str = 'LINEARREG',
        max_periods: Optional[int] = None
    ):
        """
        Initialize LINEARREG indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for linear regression calculation (default: 14)
            column_name: Name for the indicator column (default: 'LINEARREG')
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
        Calculate LINEARREG value for a specific period.

        Args:
            period: IndicatorPeriod to populate with LINEARREG value
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

        # Calculate LINEARREG using TA-Lib
        linearreg_values = talib.LINEARREG(close_prices, timeperiod=self.length)

        # The last value is the LINEARREG for our period
        linearreg_value = linearreg_values[-1]

        if not np.isnan(linearreg_value):
            setattr(period, self.column_name, float(linearreg_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 14
    ) -> np.ndarray:
        """
        Compute LINEARREG values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for linear regression calculation (default: 14)

        Returns:
            NumPy array with linear regression line values
        """
        return talib.LINEARREG(prices, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export LINEARREG values as numpy array.

        Returns:
            NumPy array with LINEARREG values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest LINEARREG value.

        Returns:
            Latest LINEARREG value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_above_regression(self) -> bool:
        """
        Check if current price is above the linear regression line.

        Returns:
            True if price > regression line
        """
        if not self.periods:
            return False

        latest_regression = self.get_latest()
        if latest_regression is None:
            return False

        current_price = self.frame.periods[-1].close_price
        return current_price > latest_regression

    def is_below_regression(self) -> bool:
        """
        Check if current price is below the linear regression line.

        Returns:
            True if price < regression line
        """
        if not self.periods:
            return False

        latest_regression = self.get_latest()
        if latest_regression is None:
            return False

        current_price = self.frame.periods[-1].close_price
        return current_price < latest_regression

    def get_distance_from_regression(self) -> Optional[float]:
        """
        Get the percentage distance from regression line.

        Returns:
            Percentage distance (positive = above, negative = below) or None
        """
        if not self.periods:
            return None

        latest_regression = self.get_latest()
        if latest_regression is None:
            return None

        current_price = self.frame.periods[-1].close_price
        return ((current_price - latest_regression) / latest_regression) * 100
