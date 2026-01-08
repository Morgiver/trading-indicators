"""SMA Crossover indicator - Composite indicator example."""

from typing import Optional, TYPE_CHECKING
import numpy as np

from ..base import BaseIndicator, IndicatorPeriod

if TYPE_CHECKING:
    from trading_frame import Frame


class SMACrossover(BaseIndicator):
    """
    SMA Crossover Detector (Composite Indicator).

    Detects crossovers between fast and slow Simple Moving Averages.
    This is a composite indicator that depends on two SMA indicators
    being present in the frame.

    Signals:
    - +1: Golden Cross (fast SMA crosses above slow SMA) - Bullish
    - -1: Death Cross (fast SMA crosses below slow SMA) - Bearish
    -  0: No crossover

    This indicator demonstrates how to build composite indicators that
    depend on other indicators by reading from frame periods.

    Example:
        >>> from trading_frame import TimeFrame
        >>> from trading_indicators import SMA, SMACrossover
        >>>
        >>> frame = TimeFrame('5T', max_periods=100)
        >>>
        >>> # First add the SMA indicators
        >>> sma20 = SMA(frame=frame, period=20, column_name='SMA_20')
        >>> sma50 = SMA(frame=frame, period=50, column_name='SMA_50')
        >>>
        >>> # Then add the crossover detector
        >>> crossover = SMACrossover(
        ...     frame=frame,
        ...     fast_column='SMA_20',
        ...     slow_column='SMA_50',
        ...     column_name='SMA_CROSS'
        ... )
        >>>
        >>> # Feed candles - crossover automatically detected
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> if crossover.periods[-1].SMA_CROSS == 1:
        ...     print("Golden Cross detected!")
        >>> elif crossover.periods[-1].SMA_CROSS == -1:
        ...     print("Death Cross detected!")

    References:
        Classic technical analysis concept used in trend following systems
    """

    def __init__(
        self,
        frame: 'Frame',
        fast_column: str = 'SMA_20',
        slow_column: str = 'SMA_50',
        column_name: str = 'SMA_CROSS',
        max_periods: Optional[int] = None
    ):
        """
        Initialize SMACrossover indicator.

        Args:
            frame: Frame to bind to
            fast_column: Column name of fast SMA (default: 'SMA_20')
            slow_column: Column name of slow SMA (default: 'SMA_50')
            column_name: Name for the crossover signal column (default: 'SMA_CROSS')
            max_periods: Maximum periods to keep (default: frame's max_periods)

        Raises:
            ValueError: If fast and slow columns are the same
        """
        if fast_column == slow_column:
            raise ValueError("Fast and slow columns must be different")

        self.fast_column = fast_column
        self.slow_column = slow_column
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate crossover signal for a specific period.

        Args:
            period: IndicatorPeriod to populate with crossover signal
        """
        # Find current index
        current_idx = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                current_idx = i
                break

        if current_idx is None or current_idx == 0:
            # No crossover on first period
            setattr(period, self.column_name, 0)
            return

        # Read current values from frame periods
        curr_fast = getattr(self.frame.periods[current_idx], self.fast_column, None)
        curr_slow = getattr(self.frame.periods[current_idx], self.slow_column, None)

        # Read previous values
        prev_fast = getattr(self.frame.periods[current_idx - 1], self.fast_column, None)
        prev_slow = getattr(self.frame.periods[current_idx - 1], self.slow_column, None)

        # Check if all values are available
        if None in [curr_fast, curr_slow, prev_fast, prev_slow]:
            setattr(period, self.column_name, 0)
            return

        # Detect Golden Cross (fast crosses above slow)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            setattr(period, self.column_name, 1)
            return

        # Detect Death Cross (fast crosses below slow)
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            setattr(period, self.column_name, -1)
            return

        # No crossover
        setattr(period, self.column_name, 0)

    def to_numpy(self) -> np.ndarray:
        """
        Export crossover signals as numpy array.

        Returns:
            NumPy array with values: +1 (golden), -1 (death), 0 (no crossover)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else 0
            for p in self.periods
        ])

    def get_latest(self) -> Optional[int]:
        """
        Get the latest crossover signal.

        Returns:
            +1 (golden), -1 (death), 0 (no crossover), or None
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, 0)
        return None

    def is_golden_cross(self) -> bool:
        """
        Check if latest signal is a golden cross.

        Returns:
            True if golden cross detected
        """
        return self.get_latest() == 1

    def is_death_cross(self) -> bool:
        """
        Check if latest signal is a death cross.

        Returns:
            True if death cross detected
        """
        return self.get_latest() == -1
