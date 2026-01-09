"""MA (Moving Average - generic) indicator."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class MA(BaseIndicator):
    """
    Moving Average (MA) - Generic indicator with multiple MA types.

    This is a flexible moving average indicator that supports various MA types
    through TA-Lib's MA function. It allows you to choose the type of moving
    average calculation without creating separate indicator instances.

    Supported MA Types:
    - SMA (0): Simple Moving Average
    - EMA (1): Exponential Moving Average
    - WMA (2): Weighted Moving Average
    - DEMA (3): Double Exponential Moving Average
    - TEMA (4): Triple Exponential Moving Average
    - TRIMA (5): Triangular Moving Average
    - KAMA (6): Kaufman Adaptive Moving Average
    - MAMA (7): MESA Adaptive Moving Average
    - T3 (8): Triple Exponential Moving Average T3

    Characteristics:
    - Unified interface for all MA types
    - Configurable MA type via parameter
    - Consistent API across different MA algorithms
    - Useful for strategy comparison and optimization

    Usage:
    - Price above MA: Uptrend
    - Price below MA: Downtrend
    - MA crossovers: Trading signals
    - Support/Resistance levels

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>>
        >>> # Create different MA types
        >>> ma_sma = MA(frame=frame, period=20, ma_type=0, column_name='MA_SMA_20')
        >>> ma_ema = MA(frame=frame, period=20, ma_type=1, column_name='MA_EMA_20')
        >>> ma_wma = MA(frame=frame, period=20, ma_type=2, column_name='MA_WMA_20')
        >>>
        >>> # Feed candles - all MAs update automatically
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ma_sma.periods[-1].MA_SMA_20)
        >>> print(ma_ema.periods[-1].MA_EMA_20)
    """

    # MA Type constants
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8

    MA_TYPE_NAMES = {
        0: "SMA",
        1: "EMA",
        2: "WMA",
        3: "DEMA",
        4: "TEMA",
        5: "TRIMA",
        6: "KAMA",
        7: "MAMA",
        8: "T3"
    }

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        ma_type: int = 0,
        column_name: str = 'MA',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MA indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for MA calculation (default: 20)
            ma_type: Type of moving average (default: 0 = SMA)
                    0 = SMA, 1 = EMA, 2 = WMA, 3 = DEMA, 4 = TEMA,
                    5 = TRIMA, 6 = KAMA, 7 = MAMA, 8 = T3
            column_name: Name for the indicator column (default: 'MA')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If period < 1 or ma_type not in valid range
        """
        if period < 1:
            raise ValueError("MA period must be at least 1")

        if ma_type not in range(9):
            raise ValueError(f"MA type must be 0-8, got {ma_type}")

        self.period = period
        self.ma_type = ma_type
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MA value for a specific period.

        Args:
            period: IndicatorPeriod to populate with MA value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Determine minimum periods based on MA type
        if self.ma_type in [3, 4, 8]:  # DEMA, TEMA, T3
            min_periods = self.period * 2
        elif self.ma_type == 6:  # KAMA
            min_periods = self.period + 1
        else:
            min_periods = self.period

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

        # Calculate MA using TA-Lib
        ma_values = talib.MA(prices_array, timeperiod=self.period, matype=self.ma_type)

        # The last value is the MA for our period
        ma_value = ma_values[-1]

        if not np.isnan(ma_value):
            setattr(period, self.column_name, round(float(ma_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export MA values as numpy array.

        Returns:
            NumPy array with MA values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest MA value.

        Returns:
            Latest MA value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def get_ma_type_name(self) -> str:
        """
        Get the name of the current MA type.

        Returns:
            String name of MA type (e.g., "SMA", "EMA")
        """
        return self.MA_TYPE_NAMES.get(self.ma_type, "UNKNOWN")
