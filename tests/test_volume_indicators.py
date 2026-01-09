"""Tests for volume indicators."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.trading_indicators.volume import AD, ADOSC, OBV


@pytest.fixture
def mock_frame():
    """Create a mock frame with sample OHLCV data."""
    frame = Mock()
    frame.max_periods = 100

    # Create realistic price data with volume
    base_price = 100.0
    periods = []

    for i in range(150):
        # Simulate price movement with trend and noise
        trend = i * 0.1
        noise = np.random.randn() * 2
        close = base_price + trend + noise
        high = close + abs(np.random.randn())
        low = close - abs(np.random.randn())
        open_price = close + np.random.randn() * 0.5

        # Volume increases on strong moves
        price_change = abs(close - (base_price + (i-1) * 0.1)) if i > 0 else 1
        volume = 1000 + int(price_change * 100) + np.random.randint(-100, 100)

        period = Mock()
        period.open_date = datetime.now() + timedelta(minutes=i * 5)
        period.close_date = None
        period.open_price = max(low, min(high, open_price))
        period.high_price = high
        period.low_price = low
        period.close_price = close
        period.volume = volume
        periods.append(period)

    frame.periods = periods
    frame.on = MagicMock()

    return frame


class TestAD:
    """Tests for AD (Chaikin A/D Line) indicator."""

    def test_initialization(self, mock_frame):
        """Test AD initialization."""
        ad = AD(mock_frame, column_name='AD')
        assert ad.column_name == 'AD'
        assert len(ad.periods) > 0

    def test_calculation(self, mock_frame):
        """Test AD calculation produces values."""
        ad = AD(mock_frame, column_name='AD')
        assert len(ad.periods) > 0

        # Check that some periods have AD values
        values_count = sum(1 for p in ad.periods if hasattr(p, 'AD'))
        assert values_count > 0

    def test_get_latest(self, mock_frame):
        """Test getting latest AD value."""
        ad = AD(mock_frame)
        latest = ad.get_latest()
        assert latest is None or isinstance(latest, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        ad = AD(mock_frame)
        array = ad.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(ad.periods)

    def test_accumulation_distribution(self, mock_frame):
        """Test accumulation/distribution detection."""
        ad = AD(mock_frame)

        # These should return boolean
        assert isinstance(ad.is_accumulating(), bool)
        assert isinstance(ad.is_distributing(), bool)

    def test_divergence_detection(self, mock_frame):
        """Test divergence detection."""
        ad = AD(mock_frame)

        # These should return boolean
        assert isinstance(ad.is_bullish_divergence(), bool)
        assert isinstance(ad.is_bearish_divergence(), bool)


class TestADOSC:
    """Tests for ADOSC (Chaikin A/D Oscillator) indicator."""

    def test_initialization(self, mock_frame):
        """Test ADOSC initialization."""
        adosc = ADOSC(mock_frame, fast=3, slow=10, column_name='ADOSC')
        assert adosc.fast == 3
        assert adosc.slow == 10
        assert adosc.column_name == 'ADOSC'
        assert len(adosc.periods) > 0

    def test_calculation(self, mock_frame):
        """Test ADOSC calculation produces values."""
        adosc = ADOSC(mock_frame)
        assert len(adosc.periods) > 0

        # Check that some periods have ADOSC values
        values_count = sum(1 for p in adosc.periods if hasattr(p, 'ADOSC'))
        assert values_count > 0

    def test_get_latest(self, mock_frame):
        """Test getting latest ADOSC value."""
        adosc = ADOSC(mock_frame)
        latest = adosc.get_latest()
        assert latest is None or isinstance(latest, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        adosc = ADOSC(mock_frame)
        array = adosc.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(adosc.periods)

    def test_bullish_bearish(self, mock_frame):
        """Test bullish/bearish detection."""
        adosc = ADOSC(mock_frame)

        # These should return boolean
        assert isinstance(adosc.is_bullish(), bool)
        assert isinstance(adosc.is_bearish(), bool)

    def test_crossover_detection(self, mock_frame):
        """Test crossover detection."""
        adosc = ADOSC(mock_frame)

        # These should return boolean
        assert isinstance(adosc.is_bullish_crossover(), bool)
        assert isinstance(adosc.is_bearish_crossover(), bool)

    def test_divergence_detection(self, mock_frame):
        """Test divergence detection."""
        adosc = ADOSC(mock_frame)

        # These should return boolean
        assert isinstance(adosc.is_bullish_divergence(), bool)
        assert isinstance(adosc.is_bearish_divergence(), bool)


class TestOBV:
    """Tests for OBV (On Balance Volume) indicator."""

    def test_initialization(self, mock_frame):
        """Test OBV initialization."""
        obv = OBV(mock_frame, column_name='OBV')
        assert obv.column_name == 'OBV'
        assert len(obv.periods) > 0

    def test_calculation(self, mock_frame):
        """Test OBV calculation produces values."""
        obv = OBV(mock_frame)
        assert len(obv.periods) > 0

        # Check that some periods have OBV values
        values_count = sum(1 for p in obv.periods if hasattr(p, 'OBV'))
        assert values_count > 0

    def test_get_latest(self, mock_frame):
        """Test getting latest OBV value."""
        obv = OBV(mock_frame)
        latest = obv.get_latest()
        assert latest is None or isinstance(latest, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        obv = OBV(mock_frame)
        array = obv.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(obv.periods)

    def test_rising_falling(self, mock_frame):
        """Test rising/falling detection."""
        obv = OBV(mock_frame)

        # These should return boolean
        assert isinstance(obv.is_rising(), bool)
        assert isinstance(obv.is_falling(), bool)

    def test_divergence_detection(self, mock_frame):
        """Test divergence detection."""
        obv = OBV(mock_frame)

        # These should return boolean
        assert isinstance(obv.is_bullish_divergence(), bool)
        assert isinstance(obv.is_bearish_divergence(), bool)

    def test_trend_strength(self, mock_frame):
        """Test trend strength calculation."""
        obv = OBV(mock_frame)
        strength = obv.get_trend_strength(lookback=10)

        # Should return a float percentage or None
        assert strength is None or isinstance(strength, float)

    def test_trend_confirmation(self, mock_frame):
        """Test trend confirmation."""
        obv = OBV(mock_frame)

        # Should return boolean
        assert isinstance(obv.confirms_trend(), bool)


def test_all_volume_indicators_export():
    """Test that all volume indicators are exported."""
    from src.trading_indicators import volume

    expected_indicators = ['AD', 'ADOSC', 'OBV']

    assert set(volume.__all__) == set(expected_indicators)
    assert len(volume.__all__) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
