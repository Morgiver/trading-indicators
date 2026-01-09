"""DEMA (Double Exponential Moving Average) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class DEMA(BaseIndicator):
    """
    Double Exponential Moving Average (DEMA) indicator.

    DEMA is a composite of a single EMA and a double EMA that provides less lag
    than either of the two original EMAs. It was developed by Patrick Mulloy.

    Formula: DEMA = 2 × EMA(n) - EMA(EMA(n))
    where n is the time period

    Characteristics:
    - Significantly less lag than standard EMA
    - More responsive to price changes than SMA and EMA
    - Smoother than EMA while maintaining responsiveness
    - Reduces lag by using the difference between single and double EMA
    - Common periods: 9, 21, 50

    Usage:
    - Price above DEMA: Uptrend
    - Price below DEMA: Downtrend
    - DEMA crossovers: Trading signals (faster than EMA crossovers)
    - Support/Resistance: Price often bounces off DEMA levels
    - Better for short-term trading due to reduced lag

    Advantages over EMA:
    - Reacts even faster to price changes
    - Less whipsaw in choppy markets
    - Better trend identification with less lag

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> dema21 = DEMA(frame=frame, period=21, column_name='DEMA_21')
        >>>
        >>> # Feed candles - DEMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(dema21.periods[-1].DEMA_21)
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 21,
        column_name: str = 'DEMA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize DEMA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for DEMA calculation (default: 21)
                   Common values: 9, 21, 50
            column_name: Name for the indicator column (default: 'DEMA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 2
        """
        if period < 2:
            raise ValueError("DEMA period must be at least 2")

        self.period = period
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate DEMA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with DEMA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need enough periods for DEMA calculation (approximately 2 * period)
        min_periods = self.period * 2
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

        # Calculate DEMA using TA-Lib
        dema_values = talib.DEMA(prices_array, timeperiod=self.period)

        # The last value is the DEMA for our period
        dema_value = dema_values[-1]

        if not np.isnan(dema_value):
            setattr(period, self.column_name, round(float(dema_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export DEMA values as numpy array.

        Returns:
            NumPy array with DEMA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest DEMA value.

        Returns:
            Latest DEMA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
