"""MFI (Money Flow Index) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MFI(BaseIndicator):
    """
    MFI (Money Flow Index) indicator.

    MFI is a volume-weighted RSI that measures buying and selling pressure.
    It ranges from 0 to 100.

    Typical interpretation:
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    - MFI crossing above 20: Buy signal
    - MFI crossing below 80: Sell signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> mfi = MFI(frame=frame, length=14, column_name='MFI_14')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(mfi.periods[-1].MFI_14)
        >>> print(mfi.is_overbought())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'MFI',
        max_periods: Optional[int] = None
    ):
        """Initialize MFI indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate MFI value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]], dtype=np.float64)
        volumes = np.array([float(p.volume) for p in self.frame.periods[:period_index + 1]], dtype=np.float64)

        mfi_values = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=self.length)
        mfi_value = mfi_values[-1]

        if not np.isnan(mfi_value):
            setattr(period, self.column_name, float(mfi_value))

    def to_numpy(self) -> np.ndarray:
        """Export MFI values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """Export normalized MFI values (0-100 → 0-1)."""
        return self.to_numpy() / 100.0

    def get_latest(self) -> Optional[float]:
        """Get the latest MFI value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = 80.0) -> bool:
        """Check if MFI indicates overbought condition."""
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = 20.0) -> bool:
        """Check if MFI indicates oversold condition."""
        latest = self.get_latest()
        return latest is not None and latest < threshold
