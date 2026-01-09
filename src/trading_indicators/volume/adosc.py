"""ADOSC (Chaikin A/D Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ADOSC(BaseIndicator):
    """
    ADOSC (Chaikin Accumulation/Distribution Oscillator) indicator.

    The A/D Oscillator is the difference between a 3-day EMA and a 10-day EMA
    of the Accumulation/Distribution Line. It measures the momentum of the
    Accumulation/Distribution Line.

    Formula:
    - ADOSC = EMA(AD, fast) - EMA(AD, slow)

    Typical interpretation:
    - ADOSC > 0: Buying pressure dominant
    - ADOSC < 0: Selling pressure dominant
    - ADOSC crossing above 0: Bullish signal
    - ADOSC crossing below 0: Bearish signal
    - Divergence with price: Potential reversal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> adosc = ADOSC(frame=frame, fast=3, slow=10, column_name='ADOSC')
        >>>
        >>> # Feed candles - ADOSC automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(adosc.periods[-1].ADOSC)
        >>> print(adosc.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        fast: int = 3,
        slow: int = 10,
        column_name: str = 'ADOSC',
        max_periods: Optional[int] = None
    ):
        """
        Initialize ADOSC indicator.

        Args:
            frame: Frame to bind to
            fast: Fast EMA period (default: 3)
            slow: Slow EMA period (default: 10)
            column_name: Name for the indicator column (default: 'ADOSC')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.fast = fast
        self.slow = slow
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate ADOSC value for a specific period.

        Args:
            period: IndicatorPeriod to populate with ADOSC value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'slow' periods for ADOSC calculation
        if period_index is None or len(self.frame.periods) < self.slow:
            return

        # Extract OHLCV prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        volumes = np.array([float(p.volume) for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate ADOSC using TA-Lib
        adosc_values = talib.ADOSC(high_prices, low_prices, close_prices, volumes,
                                    fastperiod=self.fast, slowperiod=self.slow)

        # The last value is the ADOSC for our period
        adosc_value = adosc_values[-1]

        if not np.isnan(adosc_value):
            setattr(period, self.column_name, float(adosc_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export ADOSC values as numpy array.

        Returns:
            NumPy array with ADOSC values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest ADOSC value.

        Returns:
            Latest ADOSC value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """
        Check if ADOSC indicates bullish condition (ADOSC > 0).

        Returns:
            True if ADOSC is positive (buying pressure)
        """
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """
        Check if ADOSC indicates bearish condition (ADOSC < 0).

        Returns:
            True if ADOSC is negative (selling pressure)
        """
        latest = self.get_latest()
        return latest is not None and latest < 0

    def is_bullish_crossover(self) -> bool:
        """
        Check if ADOSC crossed above zero (bullish signal).

        Returns:
            True if ADOSC crossed above zero
        """
        if len(self.periods) < 2:
            return False

        prev_adosc = getattr(self.periods[-2], self.column_name, None)
        curr_adosc = getattr(self.periods[-1], self.column_name, None)

        if prev_adosc is not None and curr_adosc is not None:
            return prev_adosc <= 0 and curr_adosc > 0
        return False

    def is_bearish_crossover(self) -> bool:
        """
        Check if ADOSC crossed below zero (bearish signal).

        Returns:
            True if ADOSC crossed below zero
        """
        if len(self.periods) < 2:
            return False

        prev_adosc = getattr(self.periods[-2], self.column_name, None)
        curr_adosc = getattr(self.periods[-1], self.column_name, None)

        if prev_adosc is not None and curr_adosc is not None:
            return prev_adosc >= 0 and curr_adosc < 0
        return False

    def is_bullish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bullish divergence (price falling, ADOSC rising).

        Args:
            lookback: Number of periods to look back (default: 10)

        Returns:
            True if bullish divergence detected
        """
        if len(self.periods) < lookback or len(self.frame.periods) < lookback:
            return False

        # Check if price is falling
        old_price = self.frame.periods[-lookback].close_price
        curr_price = self.frame.periods[-1].close_price
        price_falling = curr_price < old_price

        # Check if ADOSC is rising
        old_adosc = getattr(self.periods[-lookback], self.column_name, None)
        curr_adosc = getattr(self.periods[-1], self.column_name, None)

        if old_adosc is not None and curr_adosc is not None:
            adosc_rising = curr_adosc > old_adosc
            return price_falling and adosc_rising

        return False

    def is_bearish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bearish divergence (price rising, ADOSC falling).

        Args:
            lookback: Number of periods to look back (default: 10)

        Returns:
            True if bearish divergence detected
        """
        if len(self.periods) < lookback or len(self.frame.periods) < lookback:
            return False

        # Check if price is rising
        old_price = self.frame.periods[-lookback].close_price
        curr_price = self.frame.periods[-1].close_price
        price_rising = curr_price > old_price

        # Check if ADOSC is falling
        old_adosc = getattr(self.periods[-lookback], self.column_name, None)
        curr_adosc = getattr(self.periods[-1], self.column_name, None)

        if old_adosc is not None and curr_adosc is not None:
            adosc_falling = curr_adosc < old_adosc
            return price_rising and adosc_falling

        return False
