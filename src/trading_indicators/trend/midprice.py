"""MIDPRICE (Midpoint Price over period) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class MIDPRICE(BaseIndicator):
    """
    Midpoint Price over period indicator.

    MIDPRICE calculates the midpoint between the highest high and lowest low
    over a specified time period. Unlike MIDPOINT which works on a single price
    field, MIDPRICE specifically uses the high and low prices of each candle.

    Formula: MIDPRICE = (HIGHEST(high, period) + LOWEST(low, period)) / 2

    Characteristics:
    - Measures the middle of the price range over the period
    - Uses high and low specifically (not close price)
    - Provides true range midpoint
    - Simple but effective support/resistance indicator
    - Common periods: 14, 20, 50

    Usage:
    - Dynamic support/resistance level
    - Mean reversion reference point
    - Trend identification:
      * Price above MIDPRICE: Bullish bias
      * Price below MIDPRICE: Bearish bias
    - Volatility assessment: Distance from MIDPRICE indicates strength
    - Range trading: Buy near MIDPRICE in ranges

    Comparison with MIDPOINT:
    - MIDPRICE: Uses high/low specifically → true range midpoint
    - MIDPOINT: Uses single price field → simpler calculation
    - MIDPRICE is more appropriate for range-based analysis

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> midprice = MIDPRICE(frame=frame, period=14, column_name='MIDPRICE_14')
        >>>
        >>> # Feed candles - MIDPRICE automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(midprice.periods[-1].MIDPRICE_14)
        >>>
        >>> # Use as support/resistance
        >>> current_price = frame.periods[-1].close_price
        >>> midprice_value = midprice.get_latest()
        >>> if current_price > midprice_value:
        ...     print("Price above range midpoint - bullish")
        >>> elif current_price < midprice_value:
        ...     print("Price below range midpoint - bearish")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 14,
        column_name: str = 'MIDPRICE',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MIDPRICE indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for MIDPRICE calculation (default: 14)
                   Common values: 14, 20, 50
            column_name: Name for the indicator column (default: 'MIDPRICE')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2

        Note:
            MIDPRICE always uses high and low prices, no price_field parameter needed
        """
        if period < 2:
            raise ValueError("MIDPRICE period must be at least 2")

        self.period = period
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MIDPRICE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with MIDPRICE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for MIDPRICE calculation
        if period_index is None or period_index < self.period - 1:
            return

        # Extract high and low prices
        highs = [p.high_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        lows = [p.low_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]

        highs_array = np.array(highs)
        lows_array = np.array(lows)

        # Remove NaN values (must be consistent in both arrays)
        valid_mask = ~(np.isnan(highs_array) | np.isnan(lows_array))
        highs_array = highs_array[valid_mask]
        lows_array = lows_array[valid_mask]

        if len(highs_array) < self.period or len(lows_array) < self.period:
            return

        # Calculate MIDPRICE using TA-Lib
        midprice_values = talib.MIDPRICE(highs_array, lows_array, timeperiod=self.period)

        # The last value is the MIDPRICE for our period
        midprice_value = midprice_values[-1]

        if not np.isnan(midprice_value):
            setattr(period, self.column_name, round(float(midprice_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export MIDPRICE values as numpy array.

        Returns:
            NumPy array with MIDPRICE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest MIDPRICE value.

        Returns:
            Latest MIDPRICE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
