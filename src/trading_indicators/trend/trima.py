"""TRIMA (Triangular Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class TRIMA(BaseIndicator):
    """
    Triangular Moving Average (TRIMA) indicator.

    TRIMA is a double-smoothed moving average that gives more weight to the
    middle portion of the data. It's essentially a moving average of a moving
    average, which creates a triangular weighting distribution over the period.

    Formula:
    - If period is even: TRIMA = SMA(SMA(price, period/2), period/2 + 1)
    - If period is odd: TRIMA = SMA(SMA(price, (period+1)/2), (period+1)/2)

    The name "triangular" comes from the shape of the weighting distribution,
    which peaks in the middle of the period and tapers off at both ends.

    Characteristics:
    - Very smooth due to double averaging
    - More lag than SMA, EMA, DEMA, TEMA
    - Emphasizes middle data points
    - Natural triangular weighting distribution
    - Excellent noise filtering
    - Common periods: 20, 50, 100

    Usage:
    - Price above TRIMA: Uptrend
    - Price below TRIMA: Downtrend
    - TRIMA crossovers: Trading signals (but with more lag)
    - Support/Resistance: Strong levels due to smoothness
    - Best for identifying major trends, not short-term moves
    - Excellent for filtering out market noise

    Advantages:
    - Extremely smooth
    - Natural weighting (no arbitrary coefficients)
    - Excellent noise reduction
    - Stable and reliable
    - Good for long-term trend identification

    Disadvantages:
    - Significant lag (more than SMA)
    - Slow to react to price changes
    - Not suitable for fast-moving markets
    - Late signals in trending markets

    Use Cases:
    - Long-term trend identification
    - Market structure analysis
    - Noise filtering for other indicators
    - Confirming major trend reversals
    - Position trading (not day trading)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('1H', max_periods=200)
        >>> trima50 = TRIMA(frame=frame, period=50, column_name='TRIMA_50')
        >>> trima100 = TRIMA(frame=frame, period=100, column_name='TRIMA_100')
        >>>
        >>> # Feed candles - TRIMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(f"TRIMA(50): {trima50.periods[-1].TRIMA_50}")
        >>> print(f"TRIMA(100): {trima100.periods[-1].TRIMA_100}")
        >>>
        >>> # Determine major trend
        >>> if trima50.get_latest() > trima100.get_latest():
        ...     print("Major uptrend confirmed")
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        column_name: str = 'TRIMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize TRIMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for TRIMA calculation (default: 20)
                   Common values: 20, 50, 100
            column_name: Name for the indicator column (default: 'TRIMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2
        """
        if period < 2:
            raise ValueError("TRIMA period must be at least 2")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate TRIMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with TRIMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for TRIMA calculation
        if period_index is None or period_index < self.period - 1:
            return

        # Extract prices according to the specified field
        if self.price_field == 'close':
            prices = [p.close_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'high':
            prices = [p.high_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'low':
            prices = [p.low_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        elif self.price_field == 'open':
            prices = [p.open_price for p in self.frame.periods[period_index - self.period + 1:period_index + 1]]
        else:
            return

        prices_array = np.array(prices)

        # Remove NaN values
        prices_array = prices_array[~np.isnan(prices_array)]

        if len(prices_array) < self.period:
            return

        # Calculate TRIMA using TA-Lib
        trima_values = talib.TRIMA(prices_array, timeperiod=self.period)

        # The last value is the TRIMA for our period
        trima_value = trima_values[-1]

        if not np.isnan(trima_value):
            setattr(period, self.column_name, round(float(trima_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export TRIMA values as numpy array.

        Returns:
            NumPy array with TRIMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest TRIMA value.

        Returns:
            Latest TRIMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
