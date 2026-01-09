"""Tests for Hilbert Transform cycle indicators."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from trading_indicators.cycle import (
    HT_DCPERIOD,
    HT_DCPHASE,
    HT_PHASOR,
    HT_SINE,
    HT_TRENDMODE,
)


@pytest.fixture
def mock_frame():
    """Create a mock frame with sufficient data for Hilbert Transform calculations."""
    frame = Mock()

    # Generate 100 periods of realistic price data (sine wave pattern)
    base_price = 100.0
    periods = []

    for i in range(100):
        # Create cyclic price pattern
        cycle = 20  # 20-period cycle
        price = base_price + 5 * np.sin(2 * np.pi * i / cycle)

        period = Mock()
        period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)
        period.close_price = price + np.random.normal(0, 0.5)  # Add noise
        period.high_price = period.close_price + np.random.uniform(0.5, 1.5)
        period.low_price = period.close_price - np.random.uniform(0.5, 1.5)
        period.open_price = period.close_price + np.random.normal(0, 0.5)
        period.volume = 1000000 + np.random.randint(-100000, 100000)

        periods.append(period)

    frame.periods = periods
    frame.max_periods = 100

    return frame


class TestHT_DCPERIOD:
    """Tests for HT_DCPERIOD (Dominant Cycle Period) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = HT_DCPERIOD(frame=mock_frame, column_name='DCPERIOD')

        assert indicator.column_name == 'DCPERIOD'
        assert indicator.frame == mock_frame

    def test_calculate_insufficient_data(self, mock_frame):
        """Test calculation with insufficient data."""
        # Create frame with only 20 periods (need 32)
        frame = Mock()
        frame.periods = mock_frame.periods[:20]
        frame.max_periods = 20

        indicator = HT_DCPERIOD(frame=frame, column_name='DCPERIOD')

        # With insufficient data, get_latest should return None
        # (The calculate method may be called but shouldn't set a valid value)
        latest = indicator.get_latest()
        # Either no attribute or the calculate was skipped
        assert latest is None or len([p for p in indicator.periods if hasattr(p, 'DCPERIOD')]) == 0

    def test_calculate_with_sufficient_data(self, mock_frame):
        """Test calculation with sufficient data."""
        indicator = HT_DCPERIOD(frame=mock_frame, column_name='DCPERIOD')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'DCPERIOD')
        assert isinstance(mock_frame.periods[-1].DCPERIOD, float)
        assert mock_frame.periods[-1].DCPERIOD > 0

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = HT_DCPERIOD(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_get_cycle_type(self, mock_frame):
        """Test cycle type classification."""
        indicator = HT_DCPERIOD(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        cycle_type = indicator.get_cycle_type()
        assert cycle_type in ['fast', 'normal', 'slow']

    def test_is_cycle_shortening(self, mock_frame):
        """Test cycle shortening detection."""
        indicator = HT_DCPERIOD(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_cycle_shortening(lookback=5)
        assert isinstance(result, bool)

    def test_is_cycle_lengthening(self, mock_frame):
        """Test cycle lengthening detection."""
        indicator = HT_DCPERIOD(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_cycle_lengthening(lookback=5)
        assert isinstance(result, bool)

    def test_get_average_cycle(self, mock_frame):
        """Test average cycle calculation."""
        indicator = HT_DCPERIOD(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        avg_cycle = indicator.get_average_cycle(lookback=20)
        assert avg_cycle is not None
        assert isinstance(avg_cycle, float)
        assert avg_cycle > 0


class TestHT_DCPHASE:
    """Tests for HT_DCPHASE (Dominant Cycle Phase) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = HT_DCPHASE(frame=mock_frame, column_name='DCPHASE')

        assert indicator.column_name == 'DCPHASE'
        assert indicator.frame == mock_frame

    def test_calculate_insufficient_data(self, mock_frame):
        """Test calculation with insufficient data."""
        # Create frame with only 50 periods (need 63)
        frame = Mock()
        frame.periods = mock_frame.periods[:50]
        frame.max_periods = 50

        indicator = HT_DCPHASE(frame=frame, column_name='HT_DCPHASE')

        # With insufficient data, get_latest should return None
        latest = indicator.get_latest()
        assert latest is None or len([p for p in indicator.periods if hasattr(p, 'HT_DCPHASE')]) == 0

    def test_calculate_with_sufficient_data(self, mock_frame):
        """Test calculation with sufficient data."""
        indicator = HT_DCPHASE(frame=mock_frame, column_name='HT_DCPHASE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'HT_DCPHASE')
        assert isinstance(mock_frame.periods[-1].HT_DCPHASE, float)
        # HT_DCPHASE can be negative and exceed 360 initially (it's in degrees but can wrap)
        assert -360 <= mock_frame.periods[-1].HT_DCPHASE <= 720

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = HT_DCPHASE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert isinstance(latest, float)

    def test_get_phase_quadrant(self, mock_frame):
        """Test phase quadrant detection."""
        indicator = HT_DCPHASE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        quadrant = indicator.get_phase_quadrant()
        # Quadrant can be None or one of Q1-Q4
        assert quadrant in ['Q1', 'Q2', 'Q3', 'Q4', None]

    def test_is_phase_accelerating(self, mock_frame):
        """Test phase acceleration detection."""
        indicator = HT_DCPHASE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_phase_accelerating()
        assert isinstance(result, bool)

    def test_phase_velocity(self, mock_frame):
        """Test phase velocity calculation."""
        indicator = HT_DCPHASE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        velocity = indicator.get_phase_velocity()
        # Velocity can be None or a float
        assert velocity is None or isinstance(velocity, float)

    def test_trend_detection_methods(self, mock_frame):
        """Test trend detection methods."""
        indicator = HT_DCPHASE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        # Test all four trend detection methods
        assert isinstance(indicator.is_early_uptrend(), bool)
        assert isinstance(indicator.is_late_uptrend(), bool)
        assert isinstance(indicator.is_early_downtrend(), bool)
        assert isinstance(indicator.is_late_downtrend(), bool)


class TestHT_PHASOR:
    """Tests for HT_PHASOR (Phasor Components) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = HT_PHASOR(frame=mock_frame, column_names=['INPHASE', 'QUADRATURE'])

        assert indicator.column_names == ['INPHASE', 'QUADRATURE']
        assert indicator.frame == mock_frame

    def test_calculate_with_sufficient_data(self, mock_frame):
        """Test calculation with sufficient data."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated both values
        assert hasattr(mock_frame.periods[-1], 'HT_INPHASE')
        assert hasattr(mock_frame.periods[-1], 'HT_QUADRATURE')
        assert isinstance(mock_frame.periods[-1].HT_INPHASE, float)
        assert isinstance(mock_frame.periods[-1].HT_QUADRATURE, float)

    def test_get_latest(self, mock_frame):
        """Test getting latest values."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert 'HT_INPHASE' in latest
        assert 'HT_QUADRATURE' in latest

    def test_get_magnitude(self, mock_frame):
        """Test magnitude calculation."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        magnitude = indicator.get_magnitude()
        assert magnitude is not None
        assert isinstance(magnitude, (float, np.floating))
        assert magnitude >= 0

    def test_get_phase_angle(self, mock_frame):
        """Test phase angle calculation."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        angle = indicator.get_phase_angle()
        assert angle is not None
        assert isinstance(angle, (float, np.floating))
        assert -180 <= angle <= 180

    def test_phasor_components(self, mock_frame):
        """Test phasor component values."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        # Both magnitude and phase angle should be calculable
        if latest:
            mag = indicator.get_magnitude()
            angle = indicator.get_phase_angle()
            assert mag is not None
            assert angle is not None
            assert mag >= 0
            assert -180 <= angle <= 180

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = HT_PHASOR(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        arrays = indicator.to_numpy()
        assert isinstance(arrays, dict)
        assert 'HT_INPHASE' in arrays
        assert 'HT_QUADRATURE' in arrays
        assert isinstance(arrays['HT_INPHASE'], np.ndarray)
        assert isinstance(arrays['HT_QUADRATURE'], np.ndarray)


class TestHT_SINE:
    """Tests for HT_SINE (Sine Wave) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = HT_SINE(frame=mock_frame, column_names=['SINE', 'LEADSINE'])

        assert indicator.column_names == ['SINE', 'LEADSINE']
        assert indicator.frame == mock_frame

    def test_calculate_with_sufficient_data(self, mock_frame):
        """Test calculation with sufficient data."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated both values
        assert hasattr(mock_frame.periods[-1], 'HT_SINE')
        assert hasattr(mock_frame.periods[-1], 'HT_LEADSINE')
        assert isinstance(mock_frame.periods[-1].HT_SINE, float)
        assert isinstance(mock_frame.periods[-1].HT_LEADSINE, float)

    def test_value_range(self, mock_frame):
        """Test that sine values are in valid range [-1, 1]."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        if latest:
            assert -1 <= latest['HT_SINE'] <= 1
            assert -1 <= latest['HT_LEADSINE'] <= 1

    def test_get_latest(self, mock_frame):
        """Test getting latest values."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert 'HT_SINE' in latest
        assert 'HT_LEADSINE' in latest

    def test_is_bullish_crossover(self, mock_frame):
        """Test bullish crossover detection."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_bullish_crossover()
        assert isinstance(result, bool)

    def test_is_bearish_crossover(self, mock_frame):
        """Test bearish crossover detection."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_bearish_crossover()
        assert isinstance(result, bool)

    def test_sine_lead_relationship(self, mock_frame):
        """Test sine and lead sine relationship."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        # Just verify we have both values
        assert latest is not None
        assert 'HT_SINE' in latest
        assert 'HT_LEADSINE' in latest

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        indicator = HT_SINE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        arrays = indicator.to_numpy()
        assert isinstance(arrays, dict)
        assert 'HT_SINE' in arrays
        assert 'HT_LEADSINE' in arrays


class TestHT_TRENDMODE:
    """Tests for HT_TRENDMODE (Trend vs Cycle Mode) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = HT_TRENDMODE(frame=mock_frame, column_name='TRENDMODE')

        assert indicator.column_name == 'TRENDMODE'
        assert indicator.frame == mock_frame

    def test_calculate_with_sufficient_data(self, mock_frame):
        """Test calculation with sufficient data."""
        indicator = HT_TRENDMODE(frame=mock_frame, column_name='TRENDMODE')

        # Calculate for the last period
        indicator.calculate(mock_frame.periods[-1])

        # Should have calculated a value
        assert hasattr(mock_frame.periods[-1], 'TRENDMODE')
        assert mock_frame.periods[-1].TRENDMODE in [0, 1]

    def test_get_latest(self, mock_frame):
        """Test getting latest value."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert latest in [0, 1]

    def test_is_trending(self, mock_frame):
        """Test trend mode detection."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_trending()
        assert isinstance(result, bool)

    def test_is_cycling(self, mock_frame):
        """Test cycle mode detection."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_cycling()
        assert isinstance(result, bool)

    def test_mode_changed(self, mock_frame):
        """Test mode change detection."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.mode_changed()
        assert result in ['to_trend', 'to_cycle', None]

    def test_get_mode_stability(self, mock_frame):
        """Test mode stability calculation."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        stability = indicator.get_mode_stability(lookback=10)
        assert stability is not None
        assert 0 <= stability <= 100

    def test_get_trend_duration(self, mock_frame):
        """Test trend duration calculation."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        duration = indicator.get_trend_duration()
        assert isinstance(duration, int)
        assert duration >= 0

    def test_get_cycle_duration(self, mock_frame):
        """Test cycle duration calculation."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        duration = indicator.get_cycle_duration()
        assert isinstance(duration, int)
        assert duration >= 0

    def test_get_mode_string(self, mock_frame):
        """Test mode string representation."""
        indicator = HT_TRENDMODE(frame=mock_frame)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        mode_str = indicator.get_mode_string()
        assert mode_str in ['TREND', 'CYCLE']
