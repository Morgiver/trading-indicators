"""MAVP (Moving Average with Variable Period) indicator."""

from typing import Optional, TYPE_CHECKING, List
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class MAVP(BaseIndicator):
    """
    Moving Average with Variable Period (MAVP) indicator.

    MAVP calculates a moving average where the period length can vary for each
    data point. This allows for dynamic smoothing that adapts to custom logic
    or external indicators (like volatility or cycle measurements).

    The period for each bar is specified via a periods array, allowing complete
    flexibility in the smoothing behavior.

    Characteristics:
    - Variable smoothing period per data point
    - Supports multiple MA types (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3)
    - Highly customizable and adaptive
    - Can be driven by volatility, cycle, or custom logic
    - Periods array must have same length as price data

    Usage:
    - Adaptive smoothing based on market conditions
    - Volatility-adjusted moving averages
    - Cycle-based adaptive smoothing
    - Custom dynamic trend following

    Period Strategy Examples:
    - ATR-based: Wider periods during high volatility
    - Volume-based: Adjust smoothing based on volume
    - Cycle-based: Use dominant cycle length from Hilbert Transform
    - Custom logic: Any mathematical formula based on market data

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>>
        >>> # Define variable periods (e.g., based on volatility)
        >>> # Higher volatility = longer period = more smoothing
        >>> def calculate_periods(frame):
        ...     periods = []
        ...     for i, period in enumerate(frame.periods):
        ...         # Example: period = 10 + (volatility * 20)
        ...         # For demo, use a simple pattern
        ...         period_length = 10 + (i % 20)  # 10-30 period range
        ...         periods.append(period_length)
        ...     return periods
        >>>
        >>> # Create MAVP with custom period calculator
        >>> mavp = MAVP(frame=frame, period_calculator=calculate_periods,
        ...             ma_type=1, column_name='MAVP_ADAPTIVE')
        >>>
        >>> # Feed candles - MAVP automatically updates with variable periods
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(mavp.periods[-1].MAVP_ADAPTIVE)
    """

    # MA Type constants (same as MA indicator)
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8

    def __init__(
        self,
        frame: 'Frame',
        period_calculator,
        ma_type: int = 0,
        minperiod: int = 2,
        maxperiod: int = 30,
        column_name: str = 'MAVP',
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MAVP indicator.

        Args:
            frame: Frame to bind to
            period_calculator: Function that takes frame and returns list of periods
                              Must return a list with length equal to frame.periods
            ma_type: Type of moving average (default: 0 = SMA)
                    0 = SMA, 1 = EMA, 2 = WMA, 3 = DEMA, 4 = TEMA,
                    5 = TRIMA, 6 = KAMA, 7 = MAMA, 8 = T3
            minperiod: Minimum period value (default: 2)
            maxperiod: Maximum period value (default: 30)
            column_name: Name for the indicator column (default: 'MAVP')
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If minperiod < 2 or maxperiod < minperiod
        """
        if minperiod < 2:
            raise ValueError("MAVP minperiod must be at least 2")

        if maxperiod < minperiod:
            raise ValueError("MAVP maxperiod must be >= minperiod")

        if ma_type not in range(9):
            raise ValueError(f"MAVP ma_type must be 0-8, got {ma_type}")

        self.period_calculator = period_calculator
        self.ma_type = ma_type
        self.minperiod = minperiod
        self.maxperiod = maxperiod
        self.column_name = column_name
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MAVP value for a specific period.

        Args:
            period: IndicatorPeriod to populate with MAVP value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least maxperiod for stable calculation
        if period_index is None or period_index < self.maxperiod - 1:
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
        valid_indices = ~np.isnan(prices_array)
        prices_array = prices_array[valid_indices]

        if len(prices_array) < self.maxperiod:
            return

        # Calculate periods array using the provided calculator
        try:
            periods_list = self.period_calculator(self.frame)
            if len(periods_list) != len(self.frame.periods):
                return

            # Extract periods corresponding to valid price indices
            periods_array = np.array(periods_list[:period_index + 1])[valid_indices]

            # Clamp periods to min/max range
            periods_array = np.clip(periods_array, self.minperiod, self.maxperiod)

        except Exception:
            # If period calculator fails, skip this calculation
            return

        if len(periods_array) != len(prices_array):
            return

        # Calculate MAVP using TA-Lib
        mavp_values = talib.MAVP(
            prices_array,
            periods_array,
            minperiod=self.minperiod,
            maxperiod=self.maxperiod,
            matype=self.ma_type
        )

        # The last value is the MAVP for our period
        mavp_value = mavp_values[-1]

        if not np.isnan(mavp_value):
            setattr(period, self.column_name, round(float(mavp_value), 4))

    def to_numpy(self) -> np.ndarray:
        """
        Export MAVP values as numpy array.

        Returns:
            NumPy array with MAVP values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest MAVP value.

        Returns:
            Latest MAVP value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
