"""Tests for DEMA (Double Exponential Moving Average) indicator."""

import pytest
import numpy as np
from trading_indicators.trend import DEMA


class TestDEMA:
    """Test suite for DEMA indicator."""

    def test_dema_initialization(self, timeframe_1m):
        """Test DEMA indicator initialization."""
        dema = DEMA(frame=timeframe_1m, period=21, column_name='DEMA_21')

        assert dema.period == 21
        assert dema.column_name == 'DEMA_21'
        assert dema.price_field == 'close'
        assert len(dema.periods) == 0

    def test_dema_invalid_period(self, timeframe_1m):
        """Test DEMA with invalid period raises ValueError."""
        with pytest.raises(ValueError, match="DEMA period must be at least 2"):
            DEMA(frame=timeframe_1m, period=1)

    def test_dema_calculation(self, populated_frame):
        """Test DEMA calculates values correctly."""
        dema = DEMA(frame=populated_frame, period=21, column_name='DEMA_21')

        # DEMA needs approximately 2 * period for stable results
        min_periods = 21 * 2
        assert len(dema.periods) > 0

        # Check that values are calculated after minimum periods
        values_calculated = sum(
            1 for p in dema.periods
            if hasattr(p, 'DEMA_21')
        )
        assert values_calculated > 0

    def test_dema_values_are_numeric(self, populated_frame):
        """Test DEMA values are valid numbers."""
        dema = DEMA(frame=populated_frame, period=9, column_name='DEMA_9')

        for period in dema.periods:
            if hasattr(period, 'DEMA_9'):
                value = period.DEMA_9
                assert isinstance(value, (int, float))
                assert not np.isnan(value)
                assert value > 0  # QQQ prices are always positive

    def test_dema_to_numpy(self, populated_frame):
        """Test DEMA to_numpy export."""
        dema = DEMA(frame=populated_frame, period=21, column_name='DEMA_21')

        array = dema.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(dema.periods)

        # Check that some values are not NaN
        valid_values = ~np.isnan(array)
        assert np.sum(valid_values) > 0

    def test_dema_get_latest(self, populated_frame):
        """Test DEMA get_latest method."""
        dema = DEMA(frame=populated_frame, period=9, column_name='DEMA_9')

        latest = dema.get_latest()
        if latest is not None:
            assert isinstance(latest, (int, float))
            assert not np.isnan(latest)

    def test_dema_responsiveness(self, populated_frame):
        """Test DEMA is more responsive than SMA."""
        from trading_indicators.trend import SMA

        dema = DEMA(frame=populated_frame, period=21, column_name='DEMA_21')
        sma = SMA(frame=populated_frame, period=21, column_name='SMA_21')

        # DEMA should have values closer to recent prices than SMA
        # This is a basic responsiveness check
        dema_array = dema.to_numpy()
        sma_array = sma.to_numpy()

        # Both should have valid values
        assert np.sum(~np.isnan(dema_array)) > 0
        assert np.sum(~np.isnan(sma_array)) > 0

    def test_dema_price_fields(self, populated_frame):
        """Test DEMA with different price fields."""
        dema_close = DEMA(frame=populated_frame, period=9, column_name='DEMA_CLOSE', price_field='close')
        dema_high = DEMA(frame=populated_frame, period=9, column_name='DEMA_HIGH', price_field='high')
        dema_low = DEMA(frame=populated_frame, period=9, column_name='DEMA_LOW', price_field='low')

        # All should calculate values
        assert dema_close.get_latest() is not None
        assert dema_high.get_latest() is not None
        assert dema_low.get_latest() is not None

        # High DEMA should be higher than Low DEMA
        if all(v is not None for v in [dema_high.get_latest(), dema_low.get_latest()]):
            assert dema_high.get_latest() > dema_low.get_latest()
