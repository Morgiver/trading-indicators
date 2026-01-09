"""BOP (Balance Of Power) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class BOP(BaseIndicator):
    """
    BOP (Balance Of Power) indicator.

    BOP measures the strength of buyers versus sellers by analyzing
    the relationship between open, high, low, and close prices.
    Formula: (Close - Open) / (High - Low)

    Typical interpretation:
    - BOP > 0: Buyers in control (bullish)
    - BOP < 0: Sellers in control (bearish)
    - BOP near +1: Very strong buying pressure
    - BOP near -1: Very strong selling pressure
    - BOP near 0: Balance between buyers and sellers

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> bop = BOP(frame=frame, column_name='BOP')
        >>>
        >>> # Feed candles - BOP automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(bop.periods[-1].BOP)
        >>> print(bop.is_buyers_control())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'BOP',
        max_periods: Optional[int] = None
    ):
        """
        Initialize BOP indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'BOP')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate BOP value for a specific period.

        Args:
            period: IndicatorPeriod to populate with BOP value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 1 period for BOP calculation
        if period_index is None or len(self.frame.periods) < 1:
            return

        # Extract OHLC prices
        open_prices = np.array([p.open_price for p in self.frame.periods[:period_index + 1]])
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate BOP using TA-Lib
        bop_values = talib.BOP(open_prices, high_prices, low_prices, close_prices)

        # The last value is the BOP for our period
        bop_value = bop_values[-1]

        if not np.isnan(bop_value):
            setattr(period, self.column_name, float(bop_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export BOP values as numpy array.

        Returns:
            NumPy array with BOP values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """
        Export normalized BOP values for ML (-1 to +1 → 0-1).

        Returns:
            NumPy array with normalized BOP values [0, 1]
        """
        values = self.to_numpy()
        return (values + 1.0) / 2.0

    def get_latest(self) -> Optional[float]:
        """
        Get the latest BOP value.

        Returns:
            Latest BOP value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_buyers_control(self) -> bool:
        """
        Check if buyers are in control (BOP > 0).

        Returns:
            True if BOP is positive
        """
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_sellers_control(self) -> bool:
        """
        Check if sellers are in control (BOP < 0).

        Returns:
            True if BOP is negative
        """
        latest = self.get_latest()
        return latest is not None and latest < 0

    def is_strong_buying(self, threshold: float = 0.5) -> bool:
        """
        Check if there is strong buying pressure.

        Args:
            threshold: Strong buying threshold (default: 0.5)

        Returns:
            True if BOP is above threshold
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_strong_selling(self, threshold: float = -0.5) -> bool:
        """
        Check if there is strong selling pressure.

        Args:
            threshold: Strong selling threshold (default: -0.5)

        Returns:
            True if BOP is below threshold
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold
