"""Pivot Points (Swing High/Low) indicator."""

from typing import Optional, List, Dict, TYPE_CHECKING
import numpy as np

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class PivotPoints(BaseIndicator):
    """
    Pivot Points (Swing High/Low) Detector.

    Detects local highs and lows based on a configurable number of bars
    to the left and right of the pivot candidate.

    A Swing High is confirmed when the high price of the central bar is
    greater than all highs in the surrounding bars (left and right).
    A Swing Low is confirmed when the low price of the central bar is
    lower than all lows in the surrounding bars.

    Alternation Rule:
    - Pivots must alternate between High and Low
    - If two consecutive Highs occur without a Low between them, the newer High replaces the older
    - If two consecutive Lows occur without a High between them, the newer Low replaces the older

    Lag Behavior:
    - Pivots are confirmed 'right_bars' periods after the candidate bar
    - The pivot value is assigned to the candidate bar's period, not the confirmation bar
    - This creates a natural lag of 'right_bars' periods

    Example with left_bars=5, right_bars=2:
        Index:  117  118  119  120  121  122  123  124  125
        Price:   H    |    |    |    |    C    |    |    |

        At index 125, we can confirm if bar 122 is a pivot by comparing:
        - Bar 122's high/low against bars [117-121] (left) and [123-125] (right)

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> pivot = PivotPoints(frame=frame, left_bars=5, right_bars=2)
        >>>
        >>> # Feed candles - Pivots automatically detected
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> for period in pivot.periods:
        ...     if period.PIVOT_HIGH:
        ...         print(f"Swing High at {period.open_date}: {period.PIVOT_HIGH}")
        ...     if period.PIVOT_LOW:
        ...         print(f"Swing Low at {period.open_date}: {period.PIVOT_LOW}")

    References:
        Classic technical analysis concept used in Elliott Wave and pattern recognition
    """

    def __init__(
        self,
        frame: 'Frame',
        left_bars: int = 5,
        right_bars: int = 2,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize PivotPoints indicator.

        Args:
            frame: Frame to bind to
            left_bars: Number of bars to the left of pivot candidate (default: 5)
            right_bars: Number of bars to the right of pivot candidate (default: 2)
            column_names: Names for [high, low] (default: ['PIVOT_HIGH', 'PIVOT_LOW'])
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If left_bars < 1 or right_bars < 1
        """
        if left_bars < 1:
            raise ValueError("left_bars must be at least 1")
        if right_bars < 1:
            raise ValueError("right_bars must be at least 1")

        self.left_bars = left_bars
        self.right_bars = right_bars
        self.column_names = column_names or ['PIVOT_HIGH', 'PIVOT_LOW']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [high, low]")

        # Track last confirmed pivot state for alternation rule
        self._last_pivot_type = None  # 'high' or 'low'
        self._last_pivot_index = None  # Index where last pivot was confirmed

        super().__init__(frame, max_periods)

    def _is_swing_high(self, candidate_idx: int, confirmation_idx: int) -> bool:
        """
        Check if candidate_idx is a swing high confirmed at confirmation_idx.

        Args:
            candidate_idx: Index of pivot candidate
            confirmation_idx: Current index (candidate_idx + right_bars)

        Returns:
            True if swing high detected
        """
        if candidate_idx >= len(self.frame.periods):
            return False

        candidate_high = self.frame.periods[candidate_idx].high_price
        if candidate_high is None:
            return False

        # Check left bars
        for i in range(max(0, candidate_idx - self.left_bars), candidate_idx):
            compare_high = self.frame.periods[i].high_price
            if compare_high is None:
                continue
            if compare_high >= candidate_high:
                return False

        # Check right bars
        for i in range(candidate_idx + 1, min(len(self.frame.periods), candidate_idx + self.right_bars + 1)):
            compare_high = self.frame.periods[i].high_price
            if compare_high is None:
                continue
            if compare_high >= candidate_high:
                return False

        return True

    def _is_swing_low(self, candidate_idx: int, confirmation_idx: int) -> bool:
        """
        Check if candidate_idx is a swing low confirmed at confirmation_idx.

        Args:
            candidate_idx: Index of pivot candidate
            confirmation_idx: Current index (candidate_idx + right_bars)

        Returns:
            True if swing low detected
        """
        if candidate_idx >= len(self.frame.periods):
            return False

        candidate_low = self.frame.periods[candidate_idx].low_price
        if candidate_low is None:
            return False

        # Check left bars
        for i in range(max(0, candidate_idx - self.left_bars), candidate_idx):
            compare_low = self.frame.periods[i].low_price
            if compare_low is None:
                continue
            if compare_low <= candidate_low:
                return False

        # Check right bars
        for i in range(candidate_idx + 1, min(len(self.frame.periods), candidate_idx + self.right_bars + 1)):
            compare_low = self.frame.periods[i].low_price
            if compare_low is None:
                continue
            if compare_low <= candidate_low:
                return False

        return True

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate pivot points for a specific period.

        This is called for EACH period. We check if the current period can serve
        as the RIGHT edge to confirm a pivot that occurred 'right_bars' periods earlier.

        Args:
            period: IndicatorPeriod to check/populate
        """
        # Find current index
        current_idx = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                current_idx = i
                break

        if current_idx is None:
            return

        # Check if we have enough data to look for a pivot candidate
        if current_idx >= self.left_bars + self.right_bars:
            # The candidate is 'right_bars' periods back from current index
            candidate_idx = current_idx - self.right_bars

            # Ensure candidate has enough left context
            if candidate_idx >= self.left_bars:
                # Check for swing high
                is_high = self._is_swing_high(candidate_idx, current_idx)
                # Check for swing low
                is_low = self._is_swing_low(candidate_idx, current_idx)

                # Apply alternation rule
                if is_high and is_low:
                    # Both detected (rare but possible), respect last pivot type
                    if self._last_pivot_type == 'high':
                        # Last was high, so accept the low
                        is_high = False
                    else:
                        # Last was low (or None), so accept the high
                        is_low = False

                if is_high:
                    candidate_high = self.frame.periods[candidate_idx].high_price

                    # Alternation rule: if last was also a high, replace it
                    if self._last_pivot_type == 'high' and self._last_pivot_index is not None:
                        # Clear the previous high in our indicator periods
                        for ind_period in self.periods:
                            if ind_period.open_date == self.frame.periods[self._last_pivot_index].open_date:
                                setattr(ind_period, self.column_names[0], None)
                                break

                    # Update the CANDIDATE period in indicator
                    for ind_period in self.periods:
                        if ind_period.open_date == self.frame.periods[candidate_idx].open_date:
                            setattr(ind_period, self.column_names[0], float(candidate_high))
                            break

                    self._last_pivot_type = 'high'
                    self._last_pivot_index = candidate_idx

                if is_low:
                    candidate_low = self.frame.periods[candidate_idx].low_price

                    # Alternation rule: if last was also a low, replace it
                    if self._last_pivot_type == 'low' and self._last_pivot_index is not None:
                        # Clear the previous low in our indicator periods
                        for ind_period in self.periods:
                            if ind_period.open_date == self.frame.periods[self._last_pivot_index].open_date:
                                setattr(ind_period, self.column_names[1], None)
                                break

                    # Update the CANDIDATE period in indicator
                    for ind_period in self.periods:
                        if ind_period.open_date == self.frame.periods[candidate_idx].open_date:
                            setattr(ind_period, self.column_names[1], float(candidate_low))
                            break

                    self._last_pivot_type = 'low'
                    self._last_pivot_index = candidate_idx

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export Pivot Points values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'PIVOT_HIGH', 'PIVOT_LOW'
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
        Get the latest Pivot Points values.

        Returns:
            Dictionary with latest pivot values or None if not available
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None
