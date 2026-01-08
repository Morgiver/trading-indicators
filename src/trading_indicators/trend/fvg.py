"""FVG (Fair Value Gap) indicator."""

from typing import Optional, List, Dict, TYPE_CHECKING
import numpy as np

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class FVG(BaseIndicator):
    """
    Fair Value Gap (FVG) Detector.

    A Fair Value Gap occurs when there is a price imbalance between three consecutive candles,
    leaving a "gap" where no trading occurred. These gaps often act as support/resistance zones.

    Bullish FVG (Demand Zone):
    - Candle 1 (bearish): Creates a low
    - Candle 2: The gap candle
    - Candle 3 (bullish): Low of candle 3 > High of candle 1
    - Gap range: [High of candle 1, Low of candle 3]

    Bearish FVG (Supply Zone):
    - Candle 1 (bullish): Creates a high
    - Candle 2: The gap candle
    - Candle 3 (bearish): High of candle 3 < Low of candle 1
    - Gap range: [High of candle 3, Low of candle 1]

    The FVG is assigned to candle 2 (the middle candle where the gap occurs).

    Trading Applications:
    - Price often returns to fill FVG zones (mean reversion)
    - FVGs can act as support (bullish) or resistance (bearish)
    - Strong momentum moves often leave multiple FVGs
    - Unfilled FVGs indicate strong directional bias

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> fvg = FVG(frame=frame)
        >>>
        >>> # Feed candles - FVG automatically detected
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> for period in fvg.periods:
        ...     if period.FVG_HIGH and period.FVG_LOW:
        ...         gap_type = "Bullish" if period.FVG_LOW < period.FVG_HIGH else "Bearish"
        ...         print(f"{gap_type} FVG at {period.open_date}")
        ...         print(f"  Range: [{period.FVG_LOW}, {period.FVG_HIGH}]")

    References:
        ICT (Inner Circle Trader) concepts, Smart Money Concepts (SMC)
    """

    def __init__(
        self,
        frame: 'Frame',
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize FVG indicator.

        Args:
            frame: Frame to bind to
            column_names: Names for [high, low] (default: ['FVG_HIGH', 'FVG_LOW'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.column_names = column_names or ['FVG_HIGH', 'FVG_LOW']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [high, low]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate FVG for a specific period.

        FVG is detected on candle 2 (middle candle) when comparing 3 consecutive candles.
        We check if the current period is candle 3, and if so, we detect the FVG
        on candle 2 (previous period).

        Args:
            period: IndicatorPeriod to check/populate
        """
        # Find current index
        current_idx = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                current_idx = i
                break

        if current_idx is None or current_idx < 2:
            return

        # We are candle 3 (confirmation candle)
        # Check if FVG exists between candles 1, 2, 3
        candle_1_idx = current_idx - 2  # First candle
        candle_2_idx = current_idx - 1  # Middle candle (where FVG will be marked)
        candle_3_idx = current_idx      # Current candle (confirmation)

        # Extract price data
        high_1 = self.frame.periods[candle_1_idx].high_price
        low_1 = self.frame.periods[candle_1_idx].low_price

        high_3 = self.frame.periods[candle_3_idx].high_price
        low_3 = self.frame.periods[candle_3_idx].low_price

        if any(v is None for v in [high_1, low_1, high_3, low_3]):
            return

        # Detect Bullish FVG (Demand Zone)
        # Low of candle 3 > High of candle 1
        bullish_fvg = low_3 > high_1

        # Detect Bearish FVG (Supply Zone)
        # High of candle 3 < Low of candle 1
        bearish_fvg = high_3 < low_1

        # If we detect an FVG, mark it on candle 2 (middle candle)
        if bullish_fvg or bearish_fvg:
            # Find candle 2 in our indicator periods
            for ind_period in self.periods:
                if ind_period.open_date == self.frame.periods[candle_2_idx].open_date:
                    if bullish_fvg:
                        # Bullish FVG range: [High of candle 1, Low of candle 3]
                        fvg_low = float(high_1)
                        fvg_high = float(low_3)
                        setattr(ind_period, self.column_names[0], fvg_high)
                        setattr(ind_period, self.column_names[1], fvg_low)
                    elif bearish_fvg:
                        # Bearish FVG range: [High of candle 3, Low of candle 1]
                        fvg_low = float(high_3)
                        fvg_high = float(low_1)
                        setattr(ind_period, self.column_names[0], fvg_high)
                        setattr(ind_period, self.column_names[1], fvg_low)
                    break

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export FVG values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'FVG_HIGH', 'FVG_LOW'
            (or custom column names) mapping to numpy arrays
        """
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def get_latest(self) -> Optional[Dict[str, float]]:
        """
        Get the latest FVG values.

        Returns:
            Dictionary with latest FVG values or None if not available
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_bullish_fvg(self, period_index: int = -1) -> bool:
        """
        Check if FVG at given index is bullish (demand zone).

        Args:
            period_index: Index in periods list (default: -1 for latest)

        Returns:
            True if bullish FVG detected
        """
        if not self.periods or abs(period_index) > len(self.periods):
            return False

        period = self.periods[period_index]
        high = getattr(period, self.column_names[0], None)
        low = getattr(period, self.column_names[1], None)

        if high is not None and low is not None:
            return low < high  # Bullish: low < high

        return False

    def is_bearish_fvg(self, period_index: int = -1) -> bool:
        """
        Check if FVG at given index is bearish (supply zone).

        Args:
            period_index: Index in periods list (default: -1 for latest)

        Returns:
            True if bearish FVG detected
        """
        if not self.periods or abs(period_index) > len(self.periods):
            return False

        period = self.periods[period_index]
        high = getattr(period, self.column_names[0], None)
        low = getattr(period, self.column_names[1], None)

        if high is not None and low is not None:
            return low > high  # Bearish: low > high

        return False
