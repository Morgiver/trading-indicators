"""SAREXT (Parabolic SAR - Extended) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class SAREXT(BaseIndicator):
    """
    Parabolic SAR Extended (SAREXT) indicator.

    SAREXT is an extended version of the Parabolic SAR that provides additional
    parameters for fine-tuning the indicator's behavior. It allows separate
    configuration for long and short positions, offering more flexibility than
    the standard SAR.

    The indicator appears as dots above or below price:
    - Dots below price: Uptrend (bullish)
    - Dots above price: Downtrend (bearish)

    Extended Parameters:
    - Separate start, increment, and maximum AF for long and short
    - Start AF: Initial acceleration factor when trend begins
    - Increment AF: How much AF increases with each new extreme
    - Maximum AF: Cap on acceleration factor
    - Start value: Initial SAR placement offset
    - Offset on reverse: SAR adjustment when trend reverses

    Characteristics:
    - More granular control than standard SAR
    - Can optimize separately for longs and shorts
    - Asymmetric parameter support (bullish vs bearish bias)
    - Better adaptability to specific market conditions
    - Works best in trending markets

    Usage:
    - Same as SAR but with more control:
      * SAR below price: Uptrend (go long)
      * SAR above price: Downtrend (go short)
    - Entry/Exit signals:
      * Price crosses above SAR: Buy signal
      * Price crosses below SAR: Sell signal
    - Asymmetric strategies:
      * Tighter SAR for shorts (faster exit)
      * Looser SAR for longs (let winners run)

    Advantages over SAR:
    - Fine-tune for specific assets or timeframes
    - Optimize long and short separately
    - More control over SAR behavior
    - Can reflect trading style (aggressive/conservative)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>>
        >>> # Asymmetric SAR: Tighter for shorts, looser for longs
        >>> sarext = SAREXT(
        ...     frame=frame,
        ...     startvalue=0.0,
        ...     offsetonreverse=0.0,
        ...     accelerationinitlong=0.02,
        ...     accelerationlong=0.02,
        ...     accelerationmaxlong=0.2,
        ...     accelerationinitshort=0.02,
        ...     accelerationshort=0.03,  # Faster for shorts
        ...     accelerationmaxshort=0.3,  # Higher max for shorts
        ...     column_name='SAREXT'
        ... )
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access and use like standard SAR
        >>> if sarext.is_uptrend():
        ...     print("Uptrend")
    """

    def __init__(
        self,
        frame: 'Frame',
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
        column_name: str = 'SAREXT',
        max_periods: Optional[int] = None
    ):
        """
        Initialize SAREXT indicator.

        Args:
            frame: Frame to bind to
            startvalue: Start value for SAR (default: 0.0)
                       0.0 = automatic calculation based on price
            offsetonreverse: Offset on reverse (default: 0.0)
                            Adjustment when trend reverses
            accelerationinitlong: Initial AF for long positions (default: 0.02)
            accelerationlong: AF increment for long positions (default: 0.02)
            accelerationmaxlong: Maximum AF for long positions (default: 0.2)
            accelerationinitshort: Initial AF for short positions (default: 0.02)
            accelerationshort: AF increment for short positions (default: 0.02)
            accelerationmaxshort: Maximum AF for short positions (default: 0.2)
            column_name: Name for the indicator column (default: 'SAREXT')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If parameters not in valid range
        """
        # Validate long parameters
        if not (0.0 < accelerationinitlong <= 1.0):
            raise ValueError("SAREXT accelerationinitlong must be between 0.0 and 1.0")
        if not (0.0 < accelerationlong <= 1.0):
            raise ValueError("SAREXT accelerationlong must be between 0.0 and 1.0")
        if not (0.0 < accelerationmaxlong <= 1.0):
            raise ValueError("SAREXT accelerationmaxlong must be between 0.0 and 1.0")

        # Validate short parameters
        if not (0.0 < accelerationinitshort <= 1.0):
            raise ValueError("SAREXT accelerationinitshort must be between 0.0 and 1.0")
        if not (0.0 < accelerationshort <= 1.0):
            raise ValueError("SAREXT accelerationshort must be between 0.0 and 1.0")
        if not (0.0 < accelerationmaxshort <= 1.0):
            raise ValueError("SAREXT accelerationmaxshort must be between 0.0 and 1.0")

        self.startvalue = startvalue
        self.offsetonreverse = offsetonreverse
        self.accelerationinitlong = accelerationinitlong
        self.accelerationlong = accelerationlong
        self.accelerationmaxlong = accelerationmaxlong
        self.accelerationinitshort = accelerationinitshort
        self.accelerationshort = accelerationshort
        self.accelerationmaxshort = accelerationmaxshort
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate SAREXT value for a specific period.

        Args:
            period: IndicatorPeriod to populate with SAREXT value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # SAREXT needs at least 2 periods to start
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

        # Calculate SAREXT using TA-Lib
        sarext_values = talib.SAREXT(
            highs_array,
            lows_array,
            startvalue=self.startvalue,
            offsetonreverse=self.offsetonreverse,
            accelerationinitlong=self.accelerationinitlong,
            accelerationlong=self.accelerationlong,
            accelerationmaxlong=self.accelerationmaxlong,
            accelerationinitshort=self.accelerationinitshort,
            accelerationshort=self.accelerationshort,
            accelerationmaxshort=self.accelerationmaxshort
        )

        # The last value is the SAREXT for our period
        sarext_value = sarext_values[-1]

        if not np.isnan(sarext_value):
            setattr(period, self.column_name, round(float(sarext_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export SAREXT values as numpy array.

        Returns:
            NumPy array with SAREXT values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest SAREXT value.

        Returns:
            Latest SAREXT value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_uptrend(self) -> bool:
        """
        Check if currently in uptrend (SAREXT below price).

        Returns:
            True if SAREXT is below current close price
        """
        if not self.periods or not self.frame.periods:
            return False

        sarext_value = getattr(self.periods[-1], self.column_name, None)
        close_price = self.frame.periods[-1].close_price

        if sarext_value is not None and close_price is not None:
            return sarext_value < close_price

        return False

    def is_downtrend(self) -> bool:
        """
        Check if currently in downtrend (SAREXT above price).

        Returns:
            True if SAREXT is above current close price
        """
        if not self.periods or not self.frame.periods:
            return False

        sarext_value = getattr(self.periods[-1], self.column_name, None)
        close_price = self.frame.periods[-1].close_price

        if sarext_value is not None and close_price is not None:
            return sarext_value > close_price

        return False

    def is_bullish_reversal(self) -> bool:
        """
        Detect bullish reversal (price crossed above SAREXT).

        Returns:
            True if price crossed from below to above SAREXT in the last period
        """
        if len(self.periods) < 2 or len(self.frame.periods) < 2:
            return False

        prev_sarext = getattr(self.periods[-2], self.column_name, None)
        curr_sarext = getattr(self.periods[-1], self.column_name, None)
        prev_close = self.frame.periods[-2].close_price
        curr_close = self.frame.periods[-1].close_price

        if all(v is not None for v in [prev_sarext, curr_sarext, prev_close, curr_close]):
            return prev_close <= prev_sarext and curr_close > curr_sarext

        return False

    def is_bearish_reversal(self) -> bool:
        """
        Detect bearish reversal (price crossed below SAREXT).

        Returns:
            True if price crossed from above to below SAREXT in the last period
        """
        if len(self.periods) < 2 or len(self.frame.periods) < 2:
            return False

        prev_sarext = getattr(self.periods[-2], self.column_name, None)
        curr_sarext = getattr(self.periods[-1], self.column_name, None)
        prev_close = self.frame.periods[-2].close_price
        curr_close = self.frame.periods[-1].close_price

        if all(v is not None for v in [prev_sarext, curr_sarext, prev_close, curr_close]):
            return prev_close >= prev_sarext and curr_close < curr_sarext

        return False
