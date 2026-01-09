"""MAMA (MESA Adaptive Moving Average) indicator."""

from typing import Optional, Dict, TYPE_CHECKING
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class MAMA(BaseIndicator):
    """
    MESA Adaptive Moving Average (MAMA) indicator.

    MAMA adapts to price movement based on the rate of change of the phase,
    as measured by the Hilbert Transform Discriminator. This adaptation allows
    MAMA to be fast during trending markets and slow during ranging markets.
    MAMA is always paired with FAMA (Following Adaptive Moving Average).

    Developed by John Ehlers, MAMA uses a Hilbert Transform to compute the
    dominant cycle and adapts its smoothing based on that cycle.

    Formula: Adaptive smoothing based on instantaneous period from Hilbert Transform
    - Fast Limit: Maximum allowed alpha (responsiveness during trends)
    - Slow Limit: Minimum allowed alpha (smoothness during ranges)

    Characteristics:
    - Automatically adapts to market cycle
    - Fast response during trends
    - Smooth behavior during consolidation
    - Always produces two lines: MAMA and FAMA
    - FAMA follows MAMA with additional smoothing
    - Requires minimum 32 periods for initialization
    - No fixed period parameter (cycle-adaptive)

    Usage:
    - Price above MAMA: Uptrend
    - Price below MAMA: Downtrend
    - MAMA/FAMA crossovers: Trading signals
      * MAMA crosses above FAMA: Bullish signal
      * MAMA crosses below FAMA: Bearish signal
    - Distance between MAMA and FAMA: Trend strength
      * Large distance: Strong trend
      * Small distance: Weak trend or consolidation

    Advantages:
    - Excellent cycle-based adaptation
    - Reduces lag during trends
    - Filters noise during ranges
    - No parameter optimization needed
    - Superior crossover signals

    Limitations:
    - Requires significant historical data (minimum 32 periods)
    - More complex computation
    - May produce unstable results in early periods

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=200)
        >>> mama = MAMA(frame=frame, fastlimit=0.5, slowlimit=0.05,
        ...             column_names=['MAMA', 'FAMA'])
        >>>
        >>> # Feed candles - MAMA automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values (available after 32+ periods)
        >>> if hasattr(mama.periods[-1], 'MAMA'):
        ...     print(f"MAMA: {mama.periods[-1].MAMA}")
        ...     print(f"FAMA: {mama.periods[-1].FAMA}")
        >>>
        >>> # Detect crossovers
        >>> if mama.is_bullish_crossover():
        ...     print("Bullish MAMA/FAMA crossover")
    """

    def __init__(
        self,
        frame: 'Frame',
        fastlimit: float = 0.5,
        slowlimit: float = 0.05,
        column_names: list = None,
        price_field: str = 'close',
        max_periods: Optional[int] = None
    ):
        """
        Initialize MAMA indicator.

        Args:
            frame: Frame to bind to
            fastlimit: Fast limit for alpha (default: 0.5)
                      Range: 0.01 to 0.99 (higher = more responsive)
            slowlimit: Slow limit for alpha (default: 0.05)
                      Range: 0.01 to 0.99 (lower = more smooth)
            column_names: Names for MAMA and FAMA columns (default: ['MAMA', 'FAMA'])
            price_field: Price field to use ('close', 'high', 'low', 'open') (default: 'close')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If fastlimit or slowlimit not in valid range
        """
        if not (0.01 <= fastlimit <= 0.99):
            raise ValueError("MAMA fastlimit must be between 0.01 and 0.99")

        if not (0.01 <= slowlimit <= 0.99):
            raise ValueError("MAMA slowlimit must be between 0.01 and 0.99")

        if fastlimit <= slowlimit:
            raise ValueError("MAMA fastlimit must be greater than slowlimit")

        self.fastlimit = fastlimit
        self.slowlimit = slowlimit
        self.column_names = column_names or ['MAMA', 'FAMA']
        self.price_field = price_field
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MAMA and FAMA values for a specific period.

        Args:
            period: IndicatorPeriod to populate with MAMA and FAMA values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # MAMA requires minimum 32 periods
        min_periods = 32
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

        # Calculate MAMA using TA-Lib (returns MAMA and FAMA)
        mama_values, fama_values = talib.MAMA(
            prices_array,
            fastlimit=self.fastlimit,
            slowlimit=self.slowlimit
        )

        # The last values are for our period
        mama_value = mama_values[-1]
        fama_value = fama_values[-1]

        if not np.isnan(mama_value):
            setattr(period, self.column_names[0], round(float(mama_value), 4))

        if not np.isnan(fama_value):
            setattr(period, self.column_names[1], round(float(fama_value), 4))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export MAMA and FAMA values as numpy arrays.

        Returns:
            Dictionary with 'MAMA' and 'FAMA' keys containing numpy arrays
        """
        return {
            self.column_names[0]: np.array([
                getattr(p, self.column_names[0]) if hasattr(p, self.column_names[0]) else np.nan
                for p in self.periods
            ]),
            self.column_names[1]: np.array([
                getattr(p, self.column_names[1]) if hasattr(p, self.column_names[1]) else np.nan
                for p in self.periods
            ])
        }

    def get_latest(self) -> Optional[Dict[str, float]]:
        """
        Get the latest MAMA and FAMA values.

        Returns:
            Dictionary with MAMA and FAMA values or None if not available
        """
        if self.periods:
            mama_val = getattr(self.periods[-1], self.column_names[0], None)
            fama_val = getattr(self.periods[-1], self.column_names[1], None)

            if mama_val is not None and fama_val is not None:
                return {
                    self.column_names[0]: mama_val,
                    self.column_names[1]: fama_val
                }
        return None

    def is_bullish_crossover(self) -> bool:
        """
        Detect bullish MAMA/FAMA crossover.

        Returns:
            True if MAMA crossed above FAMA in the last period
        """
        if len(self.periods) < 2:
            return False

        prev_period = self.periods[-2]
        curr_period = self.periods[-1]

        prev_mama = getattr(prev_period, self.column_names[0], None)
        prev_fama = getattr(prev_period, self.column_names[1], None)
        curr_mama = getattr(curr_period, self.column_names[0], None)
        curr_fama = getattr(curr_period, self.column_names[1], None)

        if all(v is not None for v in [prev_mama, prev_fama, curr_mama, curr_fama]):
            return prev_mama <= prev_fama and curr_mama > curr_fama

        return False

    def is_bearish_crossover(self) -> bool:
        """
        Detect bearish MAMA/FAMA crossover.

        Returns:
            True if MAMA crossed below FAMA in the last period
        """
        if len(self.periods) < 2:
            return False

        prev_period = self.periods[-2]
        curr_period = self.periods[-1]

        prev_mama = getattr(prev_period, self.column_names[0], None)
        prev_fama = getattr(prev_period, self.column_names[1], None)
        curr_mama = getattr(curr_period, self.column_names[0], None)
        curr_fama = getattr(curr_period, self.column_names[1], None)

        if all(v is not None for v in [prev_mama, prev_fama, curr_mama, curr_fama]):
            return prev_mama >= prev_fama and curr_mama < curr_fama

        return False
