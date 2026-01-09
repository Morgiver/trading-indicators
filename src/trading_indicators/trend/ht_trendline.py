"""HT_TRENDLINE (Hilbert Transform - Instantaneous Trendline) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class HT_TRENDLINE(BaseIndicator):
    """
    Hilbert Transform - Instantaneous Trendline indicator.

    The Hilbert Transform Instantaneous Trendline is a sophisticated indicator
    that uses the Hilbert Transform to compute a smoothed instantaneous trendline
    of the dominant cycle. It adapts to market conditions and provides a dynamic
    moving average that adjusts to cycle variations.

    Characteristics:
    - Adaptive smoothing based on dominant market cycle
    - Zero-lag trend identification
    - Automatically adjusts to different market conditions
    - More sophisticated than standard moving averages
    - No fixed period parameter (adapts dynamically)
    - Requires minimum 63 data points for initialization

    Usage:
    - Price above HT_TRENDLINE: Uptrend
    - Price below HT_TRENDLINE: Downtrend
    - Crossovers: Trading signals
      * Price crosses above HT_TRENDLINE: Bullish signal
      * Price crosses below HT_TRENDLINE: Bearish signal
    - Dynamic support/resistance levels

    Advantages:
    - Adapts to market volatility automatically
    - Less lag than traditional moving averages
    - Better trend identification in varying market conditions
    - No parameter optimization needed

    Limitations:
    - Requires significant historical data (minimum 63 periods)
    - More complex computation
    - May produce unstable results in early periods

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=200)
        >>> ht = HT_TRENDLINE(frame=frame, column_name='HT_TREND')
        >>>
        >>> # Feed candles - HT_TRENDLINE automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values (available after 63+ periods)
        >>> if hasattr(ht.periods[-1], 'HT_TREND'):
        ...     print(ht.periods[-1].HT_TREND)
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'HT_TRENDLINE',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize HT_TRENDLINE indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'HT_TRENDLINE')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Note:
            Requires minimum 63 periods of historical data for stable results.
        """
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate HT_TRENDLINE value for a specific period.

        Args:
            period: IndicatorPeriod to populate with HT_TRENDLINE value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # HT_TRENDLINE requires minimum 63 periods
        min_periods = 63
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

        # Calculate HT_TRENDLINE using TA-Lib
        ht_values = talib.HT_TRENDLINE(prices_array)

        # The last value is the HT_TRENDLINE for our period
        ht_value = ht_values[-1]

        if not np.isnan(ht_value):
            setattr(period, self.column_name, round(float(ht_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export HT_TRENDLINE values as numpy array.

        Returns:
            NumPy array with HT_TRENDLINE values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest HT_TRENDLINE value.

        Returns:
            Latest HT_TRENDLINE value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
