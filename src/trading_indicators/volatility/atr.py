"""ATR (Average True Range) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class ATR(BaseIndicator):
    """
    Average True Range (ATR) indicator.

    Measures market volatility by calculating the average of true ranges over a period.
    Developed by J. Welles Wilder Jr.

    True Range (TR) is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|

    ATR = Moving Average of TR over N periods (typically EMA)

    Characteristics:
    - Absolute measure of volatility (not relative like Bollinger Bands %)
    - Does not indicate price direction, only volatility
    - Higher ATR = Higher volatility
    - Lower ATR = Lower volatility
    - Always positive value
    - Measured in same units as price

    Usage:
    - Volatility assessment: High ATR = volatile market, Low ATR = quiet market
    - Position sizing: Use ATR to determine stop-loss distances
    - Breakout confirmation: High ATR often accompanies strong trends
    - Risk management: ATR-based stops (e.g., 2× ATR below entry)
    - Trend strength: Expanding ATR = strong trend, contracting ATR = weak trend

    Common Applications:
    - Stop-loss placement: Entry ± (multiplier × ATR)
    - Position sizing: Risk per trade / ATR
    - Chandelier Exit: High - (multiplier × ATR)
    - Keltner Channels: EMA ± (multiplier × ATR)

    Common Periods:
    - 14 periods (Wilder's original recommendation)
    - 7 periods (short-term, more responsive)
    - 21 periods (longer-term, smoother)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> atr = ATR(frame=frame, period=14, column_name='ATR_14')
        >>>
        >>> # Feed candles - ATR automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(atr.periods[-1].ATR_14)
        >>> # Use for stop-loss: entry_price - (2 * atr_value)

    References:
        J. Welles Wilder Jr., "New Concepts in Technical Trading Systems" (1978)
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 14,
        column_name: str = 'ATR',
        max_periods: Optional[int] = None
    ):
        """
        Initialize ATR indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for ATR calculation (default: 14)
                   Common values: 7 (short-term), 14 (standard), 21 (long-term)
            column_name: Name for the indicator column (default: 'ATR')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("ATR period must be at least 1")

        self.period = period
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate ATR value for a specific period.

        Args:
            period: IndicatorPeriod to populate with ATR value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for ATR (need previous close for TR)
        if period_index is None or period_index < self.period:
            return

        # Extract high, low, close arrays
        high_values = np.array([
            p.high_price
            for p in self.frame.periods[period_index - self.period:period_index + 1]
        ])
        low_values = np.array([
            p.low_price
            for p in self.frame.periods[period_index - self.period:period_index + 1]
        ])
        close_values = np.array([
            p.close_price
            for p in self.frame.periods[period_index - self.period:period_index + 1]
        ])

        # Remove NaN values (must have all three values for each period)
        valid_mask = ~(np.isnan(high_values) | np.isnan(low_values) | np.isnan(close_values))
        high_values = high_values[valid_mask]
        low_values = low_values[valid_mask]
        close_values = close_values[valid_mask]

        if len(high_values) < self.period + 1:
            return

        # Calculate ATR using TA-Lib
        atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=self.period)

        # The last value is the ATR for our period
        atr_value = atr_values[-1]

        if not np.isnan(atr_value):
            setattr(period, self.column_name, round(float(atr_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export ATR values as numpy array.

        Returns:
            NumPy array with ATR values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest ATR value.

        Returns:
            Latest ATR value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
