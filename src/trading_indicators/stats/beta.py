"""BETA - Beta coefficient (relative volatility)."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class BETA(BaseIndicator):
    """
    BETA - Beta coefficient measuring relative volatility.

    Measures how much an asset moves relative to another asset (typically a benchmark).
    Beta > 1: More volatile than benchmark
    Beta = 1: Moves with benchmark
    Beta < 1: Less volatile than benchmark
    Beta < 0: Moves inversely to benchmark

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame with reference series)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> # Note: You need to manually set reference prices for beta calculation
        >>> beta = BETA(frame=frame, length=30, column_name='BETA')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)

    Example (Static utility):
        >>> btc_prices = np.array([...])
        >>> eth_prices = np.array([...])
        >>> beta_values = BETA.compute(eth_prices, btc_prices, length=30)
        >>> print(f"ETH beta vs BTC: {beta_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 5,
        column_name: str = 'BETA',
        reference_series: Optional[np.ndarray] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize BETA indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for beta calculation (default: 5)
            column_name: Name for the indicator column (default: 'BETA')
            reference_series: Reference series for comparison (e.g., benchmark prices)
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
        Calculate BETA value for a specific period.

        Note: Requires reference_series to be set for meaningful calculation.

        Args:
            period: IndicatorPeriod to populate with BETA value
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

        # Calculate BETA using TA-Lib
        beta_values = talib.BETA(close_prices, reference_slice, timeperiod=self.length)

        # The last value is the BETA for our period
        beta_value = beta_values[-1]

        if not np.isnan(beta_value):
            setattr(period, self.column_name, float(beta_value))

    @staticmethod
    def compute(
        series: np.ndarray,
        reference: np.ndarray,
        length: int = 5
    ) -> np.ndarray:
        """
        Compute BETA values without frame synchronization (utility mode).

        Args:
            series: Price series to analyze (e.g., ETH prices)
            reference: Reference series (e.g., BTC prices)
            length: Period for beta calculation (default: 5)

        Returns:
            NumPy array with BETA values
        """
        return talib.BETA(series, reference, timeperiod=length)

    def to_numpy(self) -> np.ndarray:
        """
        Export BETA values as numpy array.

        Returns:
            NumPy array with BETA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest BETA value.

        Returns:
            Latest BETA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_more_volatile(self, threshold: float = 1.0) -> bool:
        """
        Check if asset is more volatile than reference (beta > threshold).

        Args:
            threshold: Beta threshold (default: 1.0)

        Returns:
            True if beta > threshold
        """
        beta = self.get_latest()
        return beta is not None and beta > threshold

    def is_less_volatile(self, threshold: float = 1.0) -> bool:
        """
        Check if asset is less volatile than reference (beta < threshold).

        Args:
            threshold: Beta threshold (default: 1.0)

        Returns:
            True if beta < threshold
        """
        beta = self.get_latest()
        return beta is not None and beta < threshold

    def is_inverse_correlation(self) -> bool:
        """
        Check if asset moves inversely to reference (negative beta).

        Returns:
            True if beta < 0
        """
        beta = self.get_latest()
        return beta is not None and beta < 0
