"""STOCHRSI (Stochastic Relative Strength Index) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class STOCHRSI(BaseIndicator):
    """
    STOCHRSI (Stochastic Relative Strength Index) indicator.

    StochRSI applies Stochastic oscillator to RSI values instead of prices.
    Produces %K and %D lines from RSI.

    Typical interpretation:
    - StochRSI > 0.8: Overbought
    - StochRSI < 0.2: Oversold
    - %K crossing above %D: Bullish signal
    - %K crossing below %D: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> stochrsi = STOCHRSI(frame=frame, length=14, fastk_period=5, fastd_period=3)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(stochrsi.periods[-1].STOCHRSI_K)
        >>> print(stochrsi.periods[-1].STOCHRSI_D)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        fastk_period: int = 5,
        fastd_period: int = 3,
        fastd_ma_type: int = 0,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize STOCHRSI indicator.

        Args:
            frame: Frame to bind to
            length: RSI period (default: 14)
            fastk_period: Fast %K period (default: 5)
            fastd_period: Fast %D period (default: 3)
            fastd_ma_type: Fast %D MA type (default: 0 = SMA)
            column_names: Names for [%K, %D] (default: ['STOCHRSI_K', 'STOCHRSI_D'])
            max_periods: Maximum periods to keep
        """
        self.length = length
        self.fastk_period = fastk_period
        self.fastd_period = fastd_period
        self.fastd_ma_type = fastd_ma_type
        self.column_names = column_names or ['STOCHRSI_K', 'STOCHRSI_D']

        if len(self.column_names) != 2:
            raise ValueError("column_names must contain exactly 2 names [%K, %D]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate STOCHRSI values for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = self.length + self.fastk_period + self.fastd_period
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        fastk, fastd = talib.STOCHRSI(
            close_prices,
            timeperiod=self.length,
            fastk_period=self.fastk_period,
            fastd_period=self.fastd_period,
            fastd_matype=self.fastd_ma_type
        )

        if not np.isnan(fastk[-1]):
            setattr(period, self.column_names[0], float(fastk[-1]))
        if not np.isnan(fastd[-1]):
            setattr(period, self.column_names[1], float(fastd[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export STOCHRSI values as dictionary of numpy arrays."""
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def to_normalize(self) -> Dict[str, np.ndarray]:
        """Export normalized STOCHRSI values (already 0-1)."""
        return self.to_numpy()

    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get the latest STOCHRSI values."""
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_overbought(self, threshold: float = 0.8) -> bool:
        """Check if STOCHRSI indicates overbought condition."""
        latest = self.get_latest()
        if latest and self.column_names[0] in latest:
            return latest[self.column_names[0]] > threshold
        return False

    def is_oversold(self, threshold: float = 0.2) -> bool:
        """Check if STOCHRSI indicates oversold condition."""
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
