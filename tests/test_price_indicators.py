"""Tests for price indicators."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from trading_indicators.price import (
    AVGPRICE,
    MEDPRICE,
    TYPPRICE,
    WCLPRICE,
)


@pytest.fixture
def mock_frame():
    """Create a mock frame with realistic OHLC data."""
    frame = Mock()

    # Generate 50 periods of realistic OHLC data
    periods = []
    base_price = 100.0

    for i in range(50):
        period = Mock()
        period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)

        # Create realistic OHLC with some variance
        open_price = base_price + np.random.normal(0, 2)
        close_price = open_price + np.random.normal(0, 3)
        high_price = max(open_price, close_price) + abs(np.random.normal(1, 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(1, 0.5))

        period.open_price = open_price
        period.close_price = close_price
        period.high_price = high_price
        period.low_price = low_price
        period.volume = 1000000 + np.random.randint(-100000, 100000)

        periods.append(period)
        base_price = close_price  # Trend continuation

    frame.periods = periods
    frame.max_periods = 50

    return frame


class TestAVGPRICE:
    """Tests for AVGPRICE (Average Price) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = AVGPRICE(frame=mock_frame, column_name='AVGPRICE')

        assert indicator.column_name == 'AVGPRICE'
        assert indicator.frame == mock_frame

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = AVGPRICE(frame=mock_frame, column_name='AVGPRICE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'AVGPRICE')
        assert isinstance(mock_frame.periods[-1].AVGPRICE, float)

        # Verify formula: (O + H + L + C) / 4
        period = mock_frame.periods[-1]
        expected = (period.open_price + period.high_price + period.low_price + period.close_price) / 4
        assert abs(period.AVGPRICE - expected) < 0.01

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = AVGPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_is_above_close(self, mock_frame):
        """Test above close detection."""
        indicator = AVGPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_above_close()
        assert isinstance(result, bool)

    def test_is_below_close(self, mock_frame):
        """Test below close detection."""
        indicator = AVGPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_below_close()
        assert isinstance(result, bool)

    def test_get_spread_from_close(self, mock_frame):
        """Test spread from close calculation."""
        indicator = AVGPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        spread = indicator.get_spread_from_close()
        assert spread is not None
        assert isinstance(spread, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = AVGPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        array = indicator.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(mock_frame.periods)


class TestMEDPRICE:
    """Tests for MEDPRICE (Median Price) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = MEDPRICE(frame=mock_frame, column_name='MEDPRICE')

        assert indicator.column_name == 'MEDPRICE'
        assert indicator.frame == mock_frame

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = MEDPRICE(frame=mock_frame, column_name='MEDPRICE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'MEDPRICE')
        assert isinstance(mock_frame.periods[-1].MEDPRICE, float)

        # Verify formula: (H + L) / 2
        period = mock_frame.periods[-1]
        expected = (period.high_price + period.low_price) / 2
        assert abs(period.MEDPRICE - expected) < 0.01

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_is_above_close(self, mock_frame):
        """Test above close detection."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_above_close()
        assert isinstance(result, bool)

    def test_is_below_close(self, mock_frame):
        """Test below close detection."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_below_close()
        assert isinstance(result, bool)

    def test_get_close_position_in_range(self, mock_frame):
        """Test close position in range calculation."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        position = indicator.get_close_position_in_range()
        assert position is not None
        assert 0 <= position <= 1

    def test_get_range_size(self, mock_frame):
        """Test range size calculation."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        range_size = indicator.get_range_size()
        assert range_size is not None
        assert range_size >= 0

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = MEDPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        array = indicator.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(mock_frame.periods)


class TestTYPPRICE:
    """Tests for TYPPRICE (Typical Price) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = TYPPRICE(frame=mock_frame, column_name='TYPPRICE')

        assert indicator.column_name == 'TYPPRICE'
        assert indicator.frame == mock_frame

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = TYPPRICE(frame=mock_frame, column_name='TYPPRICE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'TYPPRICE')
        assert isinstance(mock_frame.periods[-1].TYPPRICE, float)

        # Verify formula: (H + L + C) / 3
        period = mock_frame.periods[-1]
        expected = (period.high_price + period.low_price + period.close_price) / 3
        assert abs(period.TYPPRICE - expected) < 0.01

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_is_above_close(self, mock_frame):
        """Test above close detection."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_above_close()
        assert isinstance(result, bool)

    def test_is_below_close(self, mock_frame):
        """Test below close detection."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_below_close()
        assert isinstance(result, bool)

    def test_get_spread_from_close(self, mock_frame):
        """Test spread from close calculation."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        spread = indicator.get_spread_from_close()
        assert spread is not None
        assert isinstance(spread, float)

    def test_is_rising(self, mock_frame):
        """Test rising detection."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_rising()
        assert isinstance(result, bool)

    def test_is_falling(self, mock_frame):
        """Test falling detection."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_falling()
        assert isinstance(result, bool)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = TYPPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        array = indicator.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(mock_frame.periods)


class TestWCLPRICE:
    """Tests for WCLPRICE (Weighted Close Price) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = WCLPRICE(frame=mock_frame, column_name='WCLPRICE')

        assert indicator.column_name == 'WCLPRICE'
        assert indicator.frame == mock_frame

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = WCLPRICE(frame=mock_frame, column_name='WCLPRICE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'WCLPRICE')
        assert isinstance(mock_frame.periods[-1].WCLPRICE, float)

        # Verify formula: (H + L + 2*C) / 4
        period = mock_frame.periods[-1]
        expected = (period.high_price + period.low_price + 2 * period.close_price) / 4
        assert abs(period.WCLPRICE - expected) < 0.01

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_is_above_close(self, mock_frame):
        """Test above close detection."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_above_close()
        assert isinstance(result, bool)

    def test_is_below_close(self, mock_frame):
        """Test below close detection."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_below_close()
        assert isinstance(result, bool)

    def test_get_spread_from_close(self, mock_frame):
        """Test spread from close calculation."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        spread = indicator.get_spread_from_close()
        assert spread is not None
        assert isinstance(spread, float)

    def test_is_rising(self, mock_frame):
        """Test rising detection."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_rising()
        assert isinstance(result, bool)

    def test_is_falling(self, mock_frame):
        """Test falling detection."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_falling()
        assert isinstance(result, bool)

    def test_get_momentum(self, mock_frame):
        """Test momentum calculation."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        momentum = indicator.get_momentum()
        assert momentum is not None
        assert isinstance(momentum, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = WCLPRICE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        array = indicator.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(mock_frame.periods)
