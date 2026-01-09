"""OBV (On Balance Volume) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class OBV(BaseIndicator):
    """
    OBV (On Balance Volume) indicator.

    OBV is a cumulative volume-based indicator that adds or subtracts volume
    based on whether the price closed higher or lower than the previous close.
    It measures buying and selling pressure as a cumulative indicator.

    Formula:
    - If Close > Previous Close: OBV = Previous OBV + Volume
    - If Close < Previous Close: OBV = Previous OBV - Volume
    - If Close = Previous Close: OBV = Previous OBV

    Typical interpretation:
    - Rising OBV: Buying pressure, confirms uptrend
    - Falling OBV: Selling pressure, confirms downtrend
    - OBV moving with price: Healthy trend
    - Divergence with price: Potential reversal signal
    - OBV breaks above resistance: Bullish signal
    - OBV breaks below support: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> obv = OBV(frame=frame, column_name='OBV')
        >>>
        >>> # Feed candles - OBV automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(obv.periods[-1].OBV)
        >>> print(obv.is_rising())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'OBV',
        max_periods: Optional[int] = None
    ):
        """
        Initialize OBV indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'OBV')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate OBV value for a specific period.

        Args:
            period: IndicatorPeriod to populate with OBV value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 1 period for OBV calculation
        if period_index is None or len(self.frame.periods) < 1:
            return

        # Extract close prices and volumes
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        volumes = np.array([float(p.volume) for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate OBV using TA-Lib
        obv_values = talib.OBV(close_prices, volumes)

        # The last value is the OBV for our period
        obv_value = obv_values[-1]

        if not np.isnan(obv_value):
            setattr(period, self.column_name, float(obv_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export OBV values as numpy array.

        Returns:
            NumPy array with OBV values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest OBV value.

        Returns:
            Latest OBV value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_rising(self) -> bool:
        """
        Check if OBV is rising (buying pressure).

        Returns:
            True if OBV is rising
        """
        if len(self.periods) < 2:
            return False

        prev_obv = getattr(self.periods[-2], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if prev_obv is not None and curr_obv is not None:
            return curr_obv > prev_obv
        return False

    def is_falling(self) -> bool:
        """
        Check if OBV is falling (selling pressure).

        Returns:
            True if OBV is falling
        """
        if len(self.periods) < 2:
            return False

        prev_obv = getattr(self.periods[-2], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if prev_obv is not None and curr_obv is not None:
            return curr_obv < prev_obv
        return False

    def is_bullish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bullish divergence (price falling, OBV rising).

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

        # Check if OBV is rising
        old_obv = getattr(self.periods[-lookback], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if old_obv is not None and curr_obv is not None:
            obv_rising = curr_obv > old_obv
            return price_falling and obv_rising

        return False

    def is_bearish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bearish divergence (price rising, OBV falling).

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

        # Check if OBV is falling
        old_obv = getattr(self.periods[-lookback], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if old_obv is not None and curr_obv is not None:
            obv_falling = curr_obv < old_obv
            return price_rising and obv_falling

        return False

    def get_trend_strength(self, lookback: int = 10) -> Optional[float]:
        """
        Calculate trend strength based on OBV change rate.

        Args:
            lookback: Number of periods to look back (default: 10)

        Returns:
            Percentage change in OBV over lookback period, or None
        """
        if len(self.periods) < lookback:
            return None

        old_obv = getattr(self.periods[-lookback], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if old_obv is not None and curr_obv is not None and old_obv != 0:
            return ((curr_obv - old_obv) / abs(old_obv)) * 100.0

        return None

    def confirms_trend(self, lookback: int = 5) -> bool:
        """
        Check if OBV confirms price trend (both moving in same direction).

        Args:
            lookback: Number of periods to look back (default: 5)

        Returns:
            True if OBV and price are moving in the same direction
        """
        if len(self.periods) < lookback or len(self.frame.periods) < lookback:
            return False

        # Check price direction
        old_price = self.frame.periods[-lookback].close_price
        curr_price = self.frame.periods[-1].close_price

        # Check OBV direction
        old_obv = getattr(self.periods[-lookback], self.column_name, None)
        curr_obv = getattr(self.periods[-1], self.column_name, None)

        if old_obv is not None and curr_obv is not None:
            price_direction = curr_price > old_price
            obv_direction = curr_obv > old_obv
            return price_direction == obv_direction

        return False
