"""STOCH (Stochastic) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class STOCH(BaseIndicator):
    """
    STOCH (Stochastic) indicator.

    Stochastic oscillator compares closing price to price range over time.
    Produces %K and %D lines.

    Typical interpretation:
    - %K or %D > 80: Overbought
    - %K or %D < 20: Oversold
    - %K crossing above %D: Bullish signal
    - %K crossing below %D: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> stoch = STOCH(frame=frame, fastk_period=14, slowk_period=3, slowd_period=3)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(stoch.periods[-1].STOCH_K)
        >>> print(stoch.periods[-1].STOCH_D)
    """

    def __init__(
        self,
        frame: 'Frame',
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowk_ma_type: int = 0,
        slowd_period: int = 3,
        slowd_ma_type: int = 0,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize STOCH indicator.

        Args:
            frame: Frame to bind to
            fastk_period: Fast %K period (default: 5)
            slowk_period: Slow %K period (default: 3)
            slowk_ma_type: Slow %K MA type (default: 0 = SMA)
            slowd_period: Slow %D period (default: 3)
            slowd_ma_type: Slow %D MA type (default: 0 = SMA)
            column_names: Names for [%K, %D] (default: ['STOCH_K', 'STOCH_D'])
            max_periods: Maximum periods to keep
        """
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowk_ma_type = slowk_ma_type
        self.slowd_period = slowd_period
        self.slowd_ma_type = slowd_ma_type
        self.column_names = column_names or ['STOCH_K', 'STOCH_D']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [%K, %D]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate STOCH values for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = self.fastk_period + self.slowk_period + self.slowd_period
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        slowk, slowd = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowk_matype=self.slowk_ma_type,
            slowd_period=self.slowd_period,
            slowd_matype=self.slowd_ma_type
        )

        if not np.isnan(slowk[-1]):
            setattr(period, self.column_names[0], float(slowk[-1]))
        if not np.isnan(slowd[-1]):
            setattr(period, self.column_names[1], float(slowd[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export STOCH values as dictionary of numpy arrays."""
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def to_normalize(self) -> Dict[str, np.ndarray]:
        """Export normalized STOCH values (0-100 → 0-1)."""
        arrays = self.to_numpy()
        return {key: arr / 100.0 for key, arr in arrays.items()}

    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get the latest STOCH values."""
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_overbought(self, threshold: float = 80.0) -> bool:
        """Check if STOCH indicates overbought condition."""
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] > threshold
        return False

    def is_oversold(self, threshold: float = 20.0) -> bool:
        """Check if STOCH indicates oversold condition."""
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] < threshold
        return False

    def is_bullish_crossover(self) -> bool:
        """Check if %K crossed above %D."""
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_k = getattr(prev, self.column_names[0], None)
        prev_d = getattr(prev, self.column_names[1], None)
        curr_k = getattr(curr, self.column_names[0], None)
        curr_d = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_k, prev_d, curr_k, curr_d]):
            return prev_k <= prev_d and curr_k > curr_d
        return False

    def is_bearish_crossover(self) -> bool:
        """Check if %K crossed below %D."""
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_k = getattr(prev, self.column_names[0], None)
        prev_d = getattr(prev, self.column_names[1], None)
        curr_k = getattr(curr, self.column_names[0], None)
        curr_d = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_k, prev_d, curr_k, curr_d]):
            return prev_k >= prev_d and curr_k < curr_d
        return False
