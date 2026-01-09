"""CORREL - Pearson's Correlation Coefficient."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class CORREL(BaseIndicator):
    """
    CORREL - Pearson's Correlation Coefficient.

    Measures the linear correlation between two price series.
    Returns values between -1 and 1:
    - 1: Perfect positive correlation (move together)
    - 0: No correlation (independent)
    - -1: Perfect negative correlation (move inversely)

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame with reference series)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> # Note: You need to manually set reference prices for correlation calculation
        >>> correl = CORREL(frame=frame, length=30, column_name='CORREL')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)

    Example (Static utility):
        >>> btc_prices = np.array([...])
        >>> eth_prices = np.array([...])
        >>> correl_values = CORREL.compute(eth_prices, btc_prices, length=30)
        >>> print(f"ETH/BTC correlation: {correl_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 30,
        column_name: str = 'CORREL',
        reference_series: Optional[np.ndarray] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize CORREL indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for correlation calculation (default: 30)
            column_name: Name for the indicator column (default: 'CORREL')
            reference_series: Reference series for comparison
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.length = length
        self.column_name = column_name
        self.reference_series = reference_series

        if frame:
            super().__init__(frame, max_periods)
        else:
            self.frame = None
            self.periods = []

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate CORREL value for a specific period.

        Note: Requires reference_series to be set for meaningful calculation.

        Args:
            period: IndicatorPeriod to populate with CORREL value
        """
        if self.reference_series is None:
            return

        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Get corresponding reference series slice
        if len(self.reference_series) < len(close_prices):
            return

        reference_slice = self.reference_series[:len(close_prices)]

        # Calculate CORREL using TA-Lib
        correl_values = talib.CORREL(close_prices, reference_slice, timeperiod=self.length)

        # The last value is the CORREL for our period
        correl_value = correl_values[-1]

        if not np.isnan(correl_value):
            setattr(period, self.column_name, float(correl_value))

    @staticmethod
    def compute(
        series1: np.ndarray,
        series2: np.ndarray,
        length: int = 30
    ) -> np.ndarray:
        """
        Compute CORREL values without frame synchronization (utility mode).

        Args:
            series1: First price series
            series2: Second price series
            length: Period for correlation calculation (default: 30)

        Returns:
            NumPy array with correlation values
        """
        return talib.CORREL(series1, series2, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export CORREL values as numpy array.

        Returns:
            NumPy array with CORREL values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest CORREL value.

        Returns:
            Latest CORREL value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_strong_positive_correl(self, threshold: float = 0.7) -> bool:
        """
        Check if there's strong positive correlation.

        Args:
            threshold: Correlation threshold (default: 0.7)

        Returns:
            True if correlation > threshold
        """
        correl = self.get_latest()
        return correl is not None and correl > threshold

    def is_strong_negative_correl(self, threshold: float = -0.7) -> bool:
        """
        Check if there's strong negative correlation.

        Args:
            threshold: Correlation threshold (default: -0.7)

        Returns:
            True if correlation < threshold
        """
        correl = self.get_latest()
        return correl is not None and correl < threshold

    def is_uncorrelated(self, threshold: float = 0.3) -> bool:
        """
        Check if series are uncorrelated (correlation near zero).

        Args:
            threshold: Threshold for uncorrelated (default: 0.3)

        Returns:
            True if abs(correlation) < threshold
        """
        correl = self.get_latest()
        return correl is not None and abs(correl) < threshold
