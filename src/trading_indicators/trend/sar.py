"""SAR (Parabolic SAR) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class SAR(BaseIndicator):
    """
    Parabolic SAR (Stop and Reverse) indicator.

    The Parabolic SAR is a trend-following indicator developed by J. Welles Wilder
    that provides entry and exit points. SAR stands for "stop and reverse," which
    is the action taken when the price crosses the SAR dots.

    The indicator appears as dots above or below price:
    - Dots below price: Uptrend (bullish)
    - Dots above price: Downtrend (bearish)

    Formula:
    - Rising SAR: SAR(n) = SAR(n-1) + AF × (EP - SAR(n-1))
    - Falling SAR: SAR(n) = SAR(n-1) - AF × (SAR(n-1) - EP)
    where:
    - AF = Acceleration Factor (increases with trend continuation)
    - EP = Extreme Point (highest high or lowest low in current trend)

    Characteristics:
    - Always in the market (long or short)
    - Trailing stop-loss mechanism
    - Acceleration factor increases with trend continuation
    - Self-adjusting based on price action
    - Works best in trending markets
    - Common parameters: acceleration=0.02, maximum=0.2

    Usage:
    - Trend direction:
      * SAR below price: Uptrend (go long)
      * SAR above price: Downtrend (go short)
    - Entry/Exit signals:
      * Price crosses above SAR: Buy signal
      * Price crosses below SAR: Sell signal
    - Trailing stop-loss:
      * Use SAR as dynamic stop-loss level
      * Tighten stop as trend continues (AF increases)

    Advantages:
    - Clear visual trend representation
    - Built-in stop-loss mechanism
    - Easy to interpret
    - Excellent for trailing stops

    Limitations:
    - Performs poorly in ranging/choppy markets (many whipsaws)
    - Always in a position (no neutral state)
    - Can give late signals in strongly trending markets

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> sar = SAR(frame=frame, acceleration=0.02, maximum=0.2,
        ...           column_name='SAR')
        >>>
        >>> # Feed candles - SAR automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> sar_value = sar.periods[-1].SAR
        >>> current_price = frame.periods[-1].close_price
        >>>
        >>> # Determine trend
        >>> if sar_value < current_price:
        ...     print("Uptrend: SAR below price")
        >>> else:
        ...     print("Downtrend: SAR above price")
        >>>
        >>> # Detect reversals
        >>> if sar.is_bullish_reversal():
        ...     print("Bullish reversal: Price crossed above SAR")
    """

    def __init__(
        self,
        frame: 'Frame',
        acceleration: float = 0.02,
        maximum: float = 0.2,
        column_name: str = 'SAR',
        max_periods: Optional[int] = None
    ):
        """
        Initialize SAR indicator.

        Args:
            frame: Frame to bind to
            acceleration: Acceleration factor (default: 0.02)
                         Typical range: 0.01 to 0.2
                         Higher values = more sensitive, faster acceleration
            maximum: Maximum acceleration factor (default: 0.2)
                    Typical range: 0.1 to 0.3
                    Limits how fast SAR can approach price
            column_name: Name for the indicator column (default: 'SAR')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If acceleration or maximum not in valid range
        """
        if not (0.0 < acceleration <= 1.0):
            raise ValueError("SAR acceleration must be between 0.0 and 1.0")

        if not (0.0 < maximum <= 1.0):
            raise ValueError("SAR maximum must be between 0.0 and 1.0")

        if maximum < acceleration:
            raise ValueError("SAR maximum must be >= acceleration")

        self.acceleration = acceleration
        self.maximum = maximum
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate SAR value for a specific period.

        Args:
            period: IndicatorPeriod to populate with SAR value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # SAR needs at least 2 periods to start
        if period_index is None or period_index < 1:
            return

        # Extract high and low prices
        highs = [p.high_price for p in self.frame.periods[:period_index + 1]]
        lows = [p.low_price for p in self.frame.periods[:period_index + 1]]

        highs_array = np.array(highs)
        lows_array = np.array(lows)

        # Remove NaN values (must be consistent in both arrays)
        valid_mask = ~(np.isnan(highs_array) | np.isnan(lows_array))
        highs_array = highs_array[valid_mask]
        lows_array = lows_array[valid_mask]

        if len(highs_array) < 2 or len(lows_array) < 2:
            return

        # Calculate SAR using TA-Lib
        sar_values = talib.SAR(
            highs_array,
            lows_array,
            acceleration=self.acceleration,
            maximum=self.maximum
        )

        # The last value is the SAR for our period
        sar_value = sar_values[-1]

        if not np.isnan(sar_value):
            setattr(period, self.column_name, round(float(sar_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export SAR values as numpy array.

        Returns:
            NumPy array with SAR values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest SAR value.

        Returns:
            Latest SAR value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_uptrend(self) -> bool:
        """
        Check if currently in uptrend (SAR below price).

        Returns:
            True if SAR is below current close price
        """
        if not self.periods or not self.frame.periods:
            return False

        sar_value = getattr(self.periods[-1], self.column_name, None)
        close_price = self.frame.periods[-1].close_price

        if sar_value is not None and close_price is not None:
            return sar_value < close_price

        return False

    def is_downtrend(self) -> bool:
        """
        Check if currently in downtrend (SAR above price).

        Returns:
            True if SAR is above current close price
        """
        if not self.periods or not self.frame.periods:
            return False

        sar_value = getattr(self.periods[-1], self.column_name, None)
        close_price = self.frame.periods[-1].close_price

        if sar_value is not None and close_price is not None:
            return sar_value > close_price

        return False

    def is_bullish_reversal(self) -> bool:
        """
        Detect bullish reversal (price crossed above SAR).

        Returns:
            True if price crossed from below to above SAR in the last period
        """
        if len(self.periods) < 2 or len(self.frame.periods) < 2:
            return False

        prev_sar = getattr(self.periods[-2], self.column_name, None)
        curr_sar = getattr(self.periods[-1], self.column_name, None)
        prev_close = self.frame.periods[-2].close_price
        curr_close = self.frame.periods[-1].close_price

        if all(v is not None for v in [prev_sar, curr_sar, prev_close, curr_close]):
            return prev_close <= prev_sar and curr_close > curr_sar

        return False

    def is_bearish_reversal(self) -> bool:
        """
        Detect bearish reversal (price crossed below SAR).

        Returns:
            True if price crossed from above to below SAR in the last period
        """
        if len(self.periods) < 2 or len(self.frame.periods) < 2:
            return False

        prev_sar = getattr(self.periods[-2], self.column_name, None)
        curr_sar = getattr(self.periods[-1], self.column_name, None)
        prev_close = self.frame.periods[-2].close_price
        curr_close = self.frame.periods[-1].close_price

        if all(v is not None for v in [prev_sar, curr_sar, prev_close, curr_close]):
            return prev_close >= prev_sar and curr_close < curr_sar

        return False
