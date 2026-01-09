"""STDDEV - Standard Deviation."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class STDDEV(BaseIndicator):
    """
    STDDEV - Standard Deviation (volatility measure).

    Measures the dispersion of price values around the mean.
    Higher values indicate higher volatility, lower values indicate lower volatility.
    Used to identify periods of high/low volatility and potential trend changes.

    Can be used in two modes:
    1. Auto-synced indicator mode (requires frame)
    2. Static utility mode via compute() method

    Example (Auto-synced):
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> stddev = STDDEV(frame=frame, length=20, nbdev=1, column_name='STDDEV')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Check for high volatility
        >>> if stddev.is_high_volatility(threshold=2.0):
        ...     print("High volatility detected")

    Example (Static utility):
        >>> prices = np.array([100, 102, 98, 105, 97, 103, 99])
        >>> stddev_values = STDDEV.compute(prices, length=5, nbdev=1)
        >>> print(f"Current volatility: {stddev_values[-1]:.2f}")
    """

    def __init__(
        self,
        frame: 'Frame' = None,
        length: int = 20,
        nbdev: float = 1,
        column_name: str = 'STDDEV',
        max_periods: Optional[int] = None
    ):
        """
        Initialize STDDEV indicator.

        Args:
            frame: Frame to bind to (None for utility mode)
            length: Period for standard deviation calculation (default: 20)
            nbdev: Number of standard deviations (default: 1)
            column_name: Name for the indicator column (default: 'STDDEV')
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
        Calculate STDDEV value for a specific period.

        Args:
            period: IndicatorPeriod to populate with STDDEV value
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

        # Calculate STDDEV using TA-Lib
        stddev_values = talib.STDDEV(close_prices, timeperiod=self.length, nbdev=self.nbdev)

        # The last value is the stddev for our period
        stddev_value = stddev_values[-1]

        if not np.isnan(stddev_value):
            setattr(period, self.column_name, float(stddev_value))

    @staticmethod
    def compute(
        prices: np.ndarray,
        length: int = 20,
        nbdev: float = 1
    ) -> np.ndarray:
        """
        Compute STDDEV values without frame synchronization (utility mode).

        Args:
            prices: Price series to analyze
            length: Period for standard deviation calculation (default: 20)
            nbdev: Number of standard deviations (default: 1)

        Returns:
            NumPy array with standard deviation values
        """
        return talib.STDDEV(prices, timeperiod=length, nbdev=nbdev)

    def to_numpy(self) -> np.ndarray:
        """
        Export STDDEV values as numpy array.

        Returns:
            NumPy array with STDDEV values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest STDDEV value.

        Returns:
            Latest standard deviation value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_high_volatility(self, threshold: float = 2.0) -> bool:
        """
        Check if volatility is high (stddev > threshold).

        Args:
            threshold: Volatility threshold (default: 2.0)

        Returns:
            True if stddev > threshold
        """
        stddev = self.get_latest()
        return stddev is not None and stddev > threshold

    def is_low_volatility(self, threshold: float = 0.5) -> bool:
        """
        Check if volatility is low (stddev < threshold).

        Args:
            threshold: Volatility threshold (default: 0.5)

        Returns:
            True if stddev < threshold
        """
        stddev = self.get_latest()
        return stddev is not None and stddev < threshold

    def is_expanding(self) -> bool:
        """
        Check if volatility is expanding (current > previous).

        Returns:
            True if current stddev > previous stddev
        """
        if len(self.periods) < 2:
            return False

        current_stddev = getattr(self.periods[-1], self.column_name, None)
        previous_stddev = getattr(self.periods[-2], self.column_name, None)

        if current_stddev is None or previous_stddev is None:
            return False

        return current_stddev > previous_stddev

    def is_contracting(self) -> bool:
        """
        Check if volatility is contracting (current < previous).

        Returns:
            True if current stddev < previous stddev
        """
        if len(self.periods) < 2:
            return False

        current_stddev = getattr(self.periods[-1], self.column_name, None)
        previous_stddev = getattr(self.periods[-2], self.column_name, None)

        if current_stddev is None or previous_stddev is None:
            return False

        return current_stddev < previous_stddev
