"""KAMA (Kaufman Adaptive Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class KAMA(BaseIndicator):
    """
    Kaufman Adaptive Moving Average (KAMA) indicator.

    KAMA is designed to account for market noise and volatility. The indicator
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This adaptive approach makes KAMA more
    responsive to trend changes than standard moving averages.

    Formula: KAMA adapts smoothing constant based on Efficiency Ratio (ER)
    ER = Change / Volatility
    where:
    - Change = abs(Price - Price[n periods ago])
    - Volatility = sum of abs(Price - Price[1 period ago]) over n periods

    Characteristics:
    - Automatically adapts to market conditions
    - More responsive in trending markets
    - Smoother in ranging/choppy markets
    - Reduces whipsaws in sideways markets
    - Excellent noise filtering
    - Common period: 10 (for ER calculation)

    Usage:
    - Price above KAMA: Uptrend
    - Price below KAMA: Downtrend
    - KAMA crossovers: Trading signals
      * Price crosses above KAMA: Bullish signal
      * Price crosses below KAMA: Bearish signal
    - KAMA slope: Trend strength indicator
      * Steep slope: Strong trend
      * Flat KAMA: Range-bound market

    Advantages:
    - Excellent for filtering market noise
    - Adapts to changing market volatility
    - Fewer false signals in choppy markets
    - Responsive during strong trends
    - Self-optimizing (no parameter tweaking needed)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> kama = KAMA(frame=frame, period=10, column_name='KAMA_10')
        >>>
        >>> # Feed candles - KAMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(kama.periods[-1].KAMA_10)
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 10,
        column_name: str = 'KAMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize KAMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for ER calculation (default: 10)
                   Recommended: 10-30
            column_name: Name for the indicator column (default: 'KAMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2
        """
        if period < 2:
            raise ValueError("KAMA period must be at least 2")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate KAMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with KAMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period + 1' periods for KAMA calculation
        min_periods = self.period + 1
        if period_index is None or period_index < min_periods - 1:
            return

        # Extract prices according to the specified field
        if self.price_field == 'close':
            prices = [p.close_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'high':
            prices = [p.high_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'low':
            prices = [p.low_price for p in self.frame.periods[:period_index + 1]]
        elif self.price_field == 'open':
            prices = [p.open_price for p in self.frame.periods[:period_index + 1]]
        else:
            return

        prices_array = np.array(prices)

        # Remove NaN values
        prices_array = prices_array[~np.isnan(prices_array)]

        if len(prices_array) < min_periods:
            return

        # Calculate KAMA using TA-Lib
        kama_values = talib.KAMA(prices_array, timeperiod=self.period)

        # The last value is the KAMA for our period
        kama_value = kama_values[-1]

        if not np.isnan(kama_value):
            setattr(period, self.column_name, round(float(kama_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export KAMA values as numpy array.

        Returns:
            NumPy array with KAMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest KAMA value.

        Returns:
            Latest KAMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
