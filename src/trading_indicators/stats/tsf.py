"""TSF - Time Series Forecast."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class TSF(BaseIndicator):
    """
    TSF - Time Series Forecast using linear regression.

    Calculates the forecasted value for the next period based on linear regression
    of historical prices. Projects the regression line one period forward.
    Useful for identifying potential price targets and trend continuation.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> tsf = TSF(frame=frame, length=14, column_name='TSF')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check if price is above forecast
        >>> if tsf.is_price_above_forecast():
        ...     print("Price exceeding forecast, strong momentum")

    Example (Static utility):
        >>> prices = np.array([100, 102, 104, 106, 108, 110, 112])
        >>> forecast_values = TSF.compute(prices, length=5)
        >>> print(f"Next period forecast: {forecast_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 14,
        column_name: str = 'TSF',
        max_periods: Optional[int] = None
    ):
        """
        Initialize TSF indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for forecast calculation (default: 14)
            column_name: Name for the indicator column (default: 'TSF')
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
        Calculate TSF value for a specific period.

        Args:
            period: IndicatorPeriod to populate with TSF value
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

        # Calculate TSF using TA-Lib
        tsf_values = talib.TSF(close_prices, timeperiod=self.length)

        # The last value is the forecast for our period
        tsf_value = tsf_values[-1]

        if not np.isnan(tsf_value):
            setattr(period, self.column_name, float(tsf_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 14
    ) -> np.ndarray:
        """
        Compute TSF values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for forecast calculation (default: 14)

        Returns:
            NumPy array with forecasted values
        """
        return talib.TSF(prices, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export TSF values as numpy array.

        Returns:
            NumPy array with TSF values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest TSF value.

        Returns:
            Latest forecast value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_price_above_forecast(self) -> bool:
        """
        Check if current price is above the forecast value.

        Returns:
            True if price > forecast (stronger than expected)
        """
        if not self.periods:
            return False

        latest_forecast = self.get_latest()
        if latest_forecast is None:
            return False

        current_price = self.frame.periods[-1].close_price
        return current_price > latest_forecast

    def is_price_below_forecast(self) -> bool:
        """
        Check if current price is below the forecast value.

        Returns:
            True if price < forecast (weaker than expected)
        """
        if not self.periods:
            return False

        latest_forecast = self.get_latest()
        if latest_forecast is None:
            return False

        current_price = self.frame.periods[-1].close_price
        return current_price < latest_forecast

    def get_forecast_error(self) -> Optional[float]:
        """
        Get the percentage error between actual price and forecast.

        Returns:
            Percentage error (positive = price above forecast) or None
        """
        if not self.periods:
            return None

        latest_forecast = self.get_latest()
        if latest_forecast is None:
            return None

        current_price = self.frame.periods[-1].close_price
        return ((current_price - latest_forecast) / latest_forecast) * 100
