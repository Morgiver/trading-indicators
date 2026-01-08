"""Base classes for trading indicators."""

from typing import Optional, List, Any, TYPE_CHECKING
from datetime import datetime
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from trading_frame import Frame, Period


class IndicatorPeriod:
    """
    Période d'indicateur avec accès dynamique aux valeurs.
    Utilise le même pattern _data dict que Period dans trading-frame.

    Example:
        >>> period = IndicatorPeriod(open_date=datetime.now())
        >>> period.RSI_14 = 65.4
        >>> period.SMA_20 = 50123.5
        >>> print(period.RSI_14)  # 65.4
    """

    def __init__(self, open_date: datetime):
        """
        Initialize indicator period.

        Args:
            open_date: Opening timestamp of this period
        """
        self.open_date = open_date
        self.close_date: Optional[datetime] = None
        self._data = {}

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access for indicator values.

        Args:
            name: Attribute name (e.g., 'RSI_14', 'MACD_LINE')

        Returns:
            The value stored in _data

        Raises:
            AttributeError: If attribute doesn't exist
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in self._data:
            return self._data[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """
        Dynamic attribute assignment.

        Reserved attributes (open_date, close_date, _data) are stored normally.
        All other attributes are stored in _data dict.

        Args:
            name: Attribute name
            value: Value to store
        """
        if name in ('open_date', 'close_date', '_data'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value

    def to_dict(self) -> dict:
        """
        Export period to dictionary.

        Returns:
            Dictionary with open_date, close_date, and all indicator values
        """
        result = {
            'open_date': self.open_date.isoformat(),
            'close_date': self.close_date.isoformat() if self.close_date else None,
        }
        result.update(self._data)
        return result

    def __repr__(self) -> str:
        """String representation."""
        data_str = ', '.join(f"{k}={v}" for k, v in self._data.items())
        return f"IndicatorPeriod(open_date={self.open_date}, {data_str})"


class BaseIndicator(ABC):
    """
    Base class for all technical indicators.

    The indicator automatically subscribes to frame events and synchronizes
    its periods with the frame's periods. Each indicator period contains
    calculated values accessible as dynamic attributes.

    Subclasses must implement:
    - calculate(period): Calculate indicator value(s) for a specific period
    - to_numpy(): Export indicator values as numpy array(s)

    Example:
        >>> class MyIndicator(BaseIndicator):
        ...     def calculate(self, period):
        ...         # Access frame periods and calculate
        ...         value = some_calculation(self.frame.periods)
        ...         setattr(period, self.column_name, value)
        ...
        ...     def to_numpy(self):
        ...         return np.array([getattr(p, self.column_name) for p in self.periods])
    """

    def __init__(self, frame: 'Frame', max_periods: Optional[int] = None):
        """
        Initialize indicator and bind to a Frame.

        Args:
            frame: Frame instance to bind to (TimeFrame, or any other Frame subclass)
            max_periods: Maximum number of periods to keep (defaults to frame's max_periods)
        """
        self.frame = frame
        self.max_periods = max_periods or frame.max_periods
        self.periods: List[IndicatorPeriod] = []

        # Subscribe to frame events
        self._subscribe_to_frame()

        # Initialize with existing frame periods
        self._initialize_from_frame()

    def _subscribe_to_frame(self):
        """Subscribe to frame events for automatic synchronization."""
        self.frame.on('new_period', self._on_frame_new_period)
        self.frame.on('update', self._on_frame_update)
        self.frame.on('close', self._on_frame_close)

    def _initialize_from_frame(self):
        """Initialize indicator with existing frame periods."""
        for frame_period in self.frame.periods:
            self._create_new_period(frame_period)

    def _on_frame_new_period(self, frame: 'Frame'):
        """
        Callback: New period created in frame.

        Args:
            frame: Frame that emitted the event
        """
        if frame.periods:
            self._create_new_period(frame.periods[-1])

    def _on_frame_update(self, frame: 'Frame'):
        """
        Callback: Current period updated in frame.

        Args:
            frame: Frame that emitted the event
        """
        if frame.periods and self.periods:
            self._update_current_period(frame.periods[-1])

    def _on_frame_close(self, frame: 'Frame'):
        """
        Callback: Period closed in frame.

        Args:
            frame: Frame that emitted the event
        """
        if self.periods and len(frame.periods) >= 2:
            closed_frame_period = frame.periods[-2]
            # Find and close corresponding indicator period
            for ind_period in self.periods:
                if ind_period.open_date == closed_frame_period.open_date:
                    ind_period.close_date = closed_frame_period.close_date
                    break

    def _create_new_period(self, frame_period: 'Period'):
        """
        Create a new indicator period.

        Args:
            frame_period: Corresponding frame period
        """
        # Close previous period
        if self.periods:
            self.periods[-1].close_date = frame_period.open_date

        # Create new period
        period = IndicatorPeriod(open_date=frame_period.open_date)

        # Calculate indicator value(s) for this period
        self.calculate(period)

        self.periods.append(period)

        # Maintain max_periods limit
        if len(self.periods) > self.max_periods:
            self.periods.pop(0)

    def _update_current_period(self, frame_period: 'Period'):
        """
        Update the current indicator period.

        Args:
            frame_period: Corresponding frame period
        """
        if self.periods:
            # Recalculate indicator value(s) for this period
            self.calculate(self.periods[-1])

    @abstractmethod
    def calculate(self, period: IndicatorPeriod):
        """
        Calculate indicator value(s) for a specific period.

        This method should:
        1. Access self.frame.periods to get necessary data
        2. Calculate the indicator value(s)
        3. Assign values to period using setattr(period, column_name, value)

        Args:
            period: IndicatorPeriod to populate with calculated values
        """
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self):
        """
        Export indicator values as numpy array(s).

        Returns:
            np.ndarray for single-value indicators
            Dict[str, np.ndarray] for multi-value indicators
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(periods={len(self.periods)})"
