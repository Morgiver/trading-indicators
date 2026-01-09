"""AD (Chaikin A/D Line) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class AD(BaseIndicator):
    """
    AD (Chaikin Accumulation/Distribution Line) indicator.

    The A/D Line is a volume-based indicator designed to measure the cumulative
    flow of money into and out of a security. It uses the relationship between
    close price and the high-low range to determine if accumulation (buying) or
    distribution (selling) is occurring.

    Formula:
    - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    - Money Flow Volume = Money Flow Multiplier × Volume
    - A/D Line = Previous A/D + Current Money Flow Volume

    Typical interpretation:
    - Rising A/D Line: Accumulation (buying pressure)
    - Falling A/D Line: Distribution (selling pressure)
    - Divergence with price: Potential reversal signal
    - Confirms trend when moving in same direction as price

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ad = AD(frame=frame, column_name='AD')
        >>>
        >>> # Feed candles - AD automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(ad.periods[-1].AD)
        >>> print(ad.is_accumulating())
    """

    def __init__(
        self,
        frame: 'Frame',
        column_name: str = 'AD',
        max_periods: Optional[int] = None
    ):
        """
        Initialize AD indicator.

        Args:
            frame: Frame to bind to
            column_name: Name for the indicator column (default: 'AD')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate AD value for a specific period.

        Args:
            period: IndicatorPeriod to populate with AD value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 1 period for AD calculation
        if period_index is None or len(self.frame.periods) < 1:
            return

        # Extract OHLCV prices
        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        volumes = np.array([float(p.volume) for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        # Calculate AD using TA-Lib
        ad_values = talib.AD(high_prices, low_prices, close_prices, volumes)

        # The last value is the AD for our period
        ad_value = ad_values[-1]

        if not np.isnan(ad_value):
            setattr(period, self.column_name, float(ad_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export AD values as numpy array.

        Returns:
            NumPy array with AD values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest AD value.

        Returns:
            Latest AD value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_accumulating(self) -> bool:
        """
        Check if AD indicates accumulation (rising).

        Returns:
            True if AD is rising (accumulation/buying pressure)
        """
        if len(self.periods) < 2:
            return False

        prev_ad = getattr(self.periods[-2], self.column_name, None)
        curr_ad = getattr(self.periods[-1], self.column_name, None)

        if prev_ad is not None and curr_ad is not None:
            return curr_ad > prev_ad
        return False

    def is_distributing(self) -> bool:
        """
        Check if AD indicates distribution (falling).

        Returns:
            True if AD is falling (distribution/selling pressure)
        """
        if len(self.periods) < 2:
            return False

        prev_ad = getattr(self.periods[-2], self.column_name, None)
        curr_ad = getattr(self.periods[-1], self.column_name, None)

        if prev_ad is not None and curr_ad is not None:
            return curr_ad < prev_ad
        return False

    def is_bullish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bullish divergence (price falling, AD rising).

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

        # Check if AD is rising
        old_ad = getattr(self.periods[-lookback], self.column_name, None)
        curr_ad = getattr(self.periods[-1], self.column_name, None)

        if old_ad is not None and curr_ad is not None:
            ad_rising = curr_ad > old_ad
            return price_falling and ad_rising

        return False

    def is_bearish_divergence(self, lookback: int = 10) -> bool:
        """
        Detect bearish divergence (price rising, AD falling).

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

        # Check if AD is falling
        old_ad = getattr(self.periods[-lookback], self.column_name, None)
        curr_ad = getattr(self.periods[-1], self.column_name, None)

        if old_ad is not None and curr_ad is not None:
            ad_falling = curr_ad < old_ad
            return price_rising and ad_falling

        return False
