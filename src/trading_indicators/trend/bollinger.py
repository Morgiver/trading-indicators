"""Bollinger Bands indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.

    Bollinger Bands consist of three lines:
    - Upper Band: Middle band + (standard deviation × multiplier)
    - Middle Band: Simple moving average
    - Lower Band: Middle band - (standard deviation × multiplier)

    The bands expand during high volatility and contract during low volatility.

    Typical interpretation:
    - Price touching upper band: Potentially overbought
    - Price touching lower band: Potentially oversold
    - Narrow bands: Low volatility (potential breakout)
    - Wide bands: High volatility

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> bb = BollingerBands(frame=frame, period=20, nbdevup=2.0, nbdevdn=2.0)
        >>>
        >>> # Feed candles - BB automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(bb.periods[-1].BB_UPPER)
        >>> print(bb.periods[-1].BB_MIDDLE)
        >>> print(bb.periods[-1].BB_LOWER)
        >>>
        >>> # Check volatility
        >>> bandwidth = bb.get_bandwidth()
    """

    def __init__(
        self,
        frame: 'Frame',
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize Bollinger Bands indicator.

        Args:
            frame: Frame to bind to
            period: Number of periods for SMA calculation (default: 20)
            nbdevup: Number of standard deviations for upper band (default: 2.0)
            nbdevdn: Number of standard deviations for lower band (default: 2.0)
            column_names: Names for [upper, middle, lower] (default: ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.column_names = column_names or ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']

        if len(self.column_names) != 3:
            raise ValueError("column_names must contain exactly 3 names [upper, middle, lower]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate Bollinger Bands values for a specific period.

        Args:
            period: IndicatorPeriod to populate with BB values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'period' periods for BB calculation
        if period_index is None or period_index < self.period - 1:
            return

        # Extract close prices
        close_prices = np.array([
            p.close_price
            for p in self.frame.periods[period_index - self.period + 1:period_index + 1]
        ])

        # Calculate Bollinger Bands using TA-Lib
        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=self.period,
            nbdevup=self.nbdevup,
            nbdevdn=self.nbdevdn,
            matype=0  # SMA
        )

        # The last values are for our period
        if not np.isnan(upper[-1]):
            setattr(period, self.column_names[0], float(upper[-1]))

        if not np.isnan(middle[-1]):
            setattr(period, self.column_names[1], float(middle[-1]))

        if not np.isnan(lower[-1]):
            setattr(period, self.column_names[2], float(lower[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export Bollinger Bands values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
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
        Get the latest Bollinger Bands values.

        Returns:
            Dictionary with latest BB values or None if not available
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def get_bandwidth(self) -> Optional[float]:
        """
        Get the current bandwidth (width of the bands).

        Bandwidth = (Upper Band - Lower Band) / Middle Band

        Returns:
            Current bandwidth or None if not available
        """
        if self.periods:
            upper = getattr(self.periods[-1], self.column_names[0], None)
            middle = getattr(self.periods[-1], self.column_names[1], None)
            lower = getattr(self.periods[-1], self.column_names[2], None)

            if all(v is not None for v in [upper, middle, lower]) and middle != 0:
                return (upper - lower) / middle

        return None

    def get_percent_b(self, price: Optional[float] = None) -> Optional[float]:
        """
        Get %B indicator (price position within bands).

        %B = (Price - Lower Band) / (Upper Band - Lower Band)

        Values:
        - %B > 1: Price above upper band
        - %B = 0.5: Price at middle band
        - %B < 0: Price below lower band

        Args:
            price: Price to calculate %B for (defaults to latest close price)

        Returns:
            %B value or None if not available
        """
        if not self.periods or not self.frame.periods:
            return None

        upper = getattr(self.periods[-1], self.column_names[0], None)
        lower = getattr(self.periods[-1], self.column_names[2], None)

        if price is None:
            price = self.frame.periods[-1].close_price

        if all(v is not None for v in [upper, lower]) and upper != lower:
            return (price - lower) / (upper - lower)

        return None

    def is_squeeze(self, threshold: float = 0.02) -> bool:
        """
        Detect Bollinger Band squeeze (low volatility).

        A squeeze occurs when bandwidth is below the threshold.

        Args:
            threshold: Bandwidth threshold for squeeze detection (default: 0.02 = 2%)

        Returns:
            True if squeeze detected
        """
        bandwidth = self.get_bandwidth()
        return bandwidth is not None and bandwidth < threshold
