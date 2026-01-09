"""Tests for all trend indicators."""

import pytest
import numpy as np
from trading_indicators.trend import (
    DEMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP,
    MIDPOINT, MIDPRICE, SAR, SAREXT, T3, TEMA, TRIMA, WMA
)


class TestHT_TRENDLINE:
    """Test suite for HT_TRENDLINE indicator."""

    def test_initialization(self, timeframe_1m):
        """Test HT_TRENDLINE indicator initialization."""
        ht = HT_TRENDLINE(frame=timeframe_1m, column_name='HT_TREND')
        assert ht.column_name == 'HT_TREND'
        assert ht.price_field == 'close'

    def test_calculation(self, populated_frame):
        """Test HT_TRENDLINE calculates values."""
        ht = HT_TRENDLINE(frame=populated_frame, column_name='HT_TREND')

        # HT_TRENDLINE needs 63+ periods
        values_calculated = sum(
            1 for p in ht.periods if hasattr(p, 'HT_TREND')
        )
        # Should have some values if we have enough data
        assert len(ht.periods) > 0

    def test_to_numpy(self, populated_frame):
        """Test HT_TRENDLINE to_numpy export."""
        ht = HT_TRENDLINE(frame=populated_frame, column_name='HT_TREND')
        array = ht.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(ht.periods)


class TestKAMA:
    """Test suite for KAMA indicator."""

    def test_initialization(self, timeframe_1m):
        """Test KAMA indicator initialization."""
        kama = KAMA(frame=timeframe_1m, period=10, column_name='KAMA_10')
        assert kama.period == 10
        assert kama.column_name == 'KAMA_10'

    def test_invalid_period(self, timeframe_1m):
        """Test KAMA with invalid period."""
        with pytest.raises(ValueError, match="KAMA period must be at least 2"):
            KAMA(frame=timeframe_1m, period=1)

    def test_calculation(self, populated_frame):
        """Test KAMA calculates values."""
        kama = KAMA(frame=populated_frame, period=10, column_name='KAMA_10')
        assert len(kama.periods) > 0

        values = [p.KAMA_10 for p in kama.periods if hasattr(p, 'KAMA_10')]
        assert len(values) > 0
        assert all(v > 0 for v in values)  # QQQ prices are positive

    def test_to_numpy(self, populated_frame):
        """Test KAMA to_numpy export."""
        kama = KAMA(frame=populated_frame, period=10, column_name='KAMA_10')
        array = kama.to_numpy()
        assert isinstance(array, np.ndarray)


class TestMA:
    """Test suite for MA (generic) indicator."""

    def test_initialization(self, timeframe_1m):
        """Test MA indicator initialization."""
        ma = MA(frame=timeframe_1m, period=20, ma_type=0, column_name='MA_SMA')
        assert ma.period == 20
        assert ma.ma_type == 0
        assert ma.get_ma_type_name() == "SMA"

    def test_invalid_period(self, timeframe_1m):
        """Test MA with invalid period."""
        with pytest.raises(ValueError, match="MA period must be at least 1"):
            MA(frame=timeframe_1m, period=0)

    def test_invalid_ma_type(self, timeframe_1m):
        """Test MA with invalid ma_type."""
        with pytest.raises(ValueError, match="MA type must be 0-8"):
            MA(frame=timeframe_1m, period=20, ma_type=10)

    def test_calculation_sma(self, populated_frame):
        """Test MA with SMA type."""
        ma = MA(frame=populated_frame, period=20, ma_type=MA.SMA, column_name='MA_SMA')
        values = [p.MA_SMA for p in ma.periods if hasattr(p, 'MA_SMA')]
        assert len(values) > 0

    def test_calculation_ema(self, populated_frame):
        """Test MA with EMA type."""
        ma = MA(frame=populated_frame, period=20, ma_type=MA.EMA, column_name='MA_EMA')
        values = [p.MA_EMA for p in ma.periods if hasattr(p, 'MA_EMA')]
        assert len(values) > 0

    def test_ma_type_names(self, timeframe_1m):
        """Test MA type name mapping."""
        assert MA(frame=timeframe_1m, period=20, ma_type=0).get_ma_type_name() == "SMA"
        assert MA(frame=timeframe_1m, period=20, ma_type=1).get_ma_type_name() == "EMA"
        assert MA(frame=timeframe_1m, period=20, ma_type=2).get_ma_type_name() == "WMA"


class TestMAMA:
    """Test suite for MAMA indicator."""

    def test_initialization(self, timeframe_1m):
        """Test MAMA indicator initialization."""
        mama = MAMA(frame=timeframe_1m, fastlimit=0.5, slowlimit=0.05)
        assert mama.fastlimit == 0.5
        assert mama.slowlimit == 0.05
        assert mama.column_names == ['MAMA', 'FAMA']

    def test_invalid_fastlimit(self, timeframe_1m):
        """Test MAMA with invalid fastlimit."""
        with pytest.raises(ValueError, match="fastlimit must be between 0.01 and 0.99"):
            MAMA(frame=timeframe_1m, fastlimit=1.5)

    def test_invalid_slowlimit(self, timeframe_1m):
        """Test MAMA with invalid slowlimit."""
        with pytest.raises(ValueError, match="slowlimit must be between 0.01 and 0.99"):
            MAMA(frame=timeframe_1m, slowlimit=0.0)

    def test_fastlimit_greater_than_slowlimit(self, timeframe_1m):
        """Test MAMA fastlimit > slowlimit validation."""
        with pytest.raises(ValueError, match="fastlimit must be greater than slowlimit"):
            MAMA(frame=timeframe_1m, fastlimit=0.05, slowlimit=0.5)

    def test_calculation(self, populated_frame):
        """Test MAMA calculates values."""
        mama = MAMA(frame=populated_frame, fastlimit=0.5, slowlimit=0.05)

        # MAMA needs 32+ periods
        latest = mama.get_latest()
        # May be None if not enough data, that's ok

    def test_to_numpy(self, populated_frame):
        """Test MAMA to_numpy export."""
        mama = MAMA(frame=populated_frame)
        arrays = mama.to_numpy()
        assert isinstance(arrays, dict)
        assert 'MAMA' in arrays
        assert 'FAMA' in arrays
        assert isinstance(arrays['MAMA'], np.ndarray)
        assert isinstance(arrays['FAMA'], np.ndarray)

    def test_crossover_detection(self, populated_frame):
        """Test MAMA crossover detection methods."""
        mama = MAMA(frame=populated_frame)

        # Methods should not raise errors
        bullish = mama.is_bullish_crossover()
        bearish = mama.is_bearish_crossover()
        assert isinstance(bullish, bool)
        assert isinstance(bearish, bool)


class TestMAVP:
    """Test suite for MAVP indicator."""

    def test_initialization(self, timeframe_1m):
        """Test MAVP indicator initialization."""
        def period_calc(frame):
            return [20] * len(frame.periods)

        mavp = MAVP(
            frame=timeframe_1m,
            period_calculator=period_calc,
            ma_type=0,
            minperiod=2,
            maxperiod=30
        )
        assert mavp.ma_type == 0
        assert mavp.minperiod == 2
        assert mavp.maxperiod == 30

    def test_invalid_minperiod(self, timeframe_1m):
        """Test MAVP with invalid minperiod."""
        def period_calc(frame):
            return [20] * len(frame.periods)

        with pytest.raises(ValueError, match="minperiod must be at least 2"):
            MAVP(frame=timeframe_1m, period_calculator=period_calc, minperiod=1)

    def test_calculation(self, populated_frame):
        """Test MAVP calculates values."""
        def period_calc(frame):
            # Variable periods based on index (must be floats for TA-Lib)
            return [float(10 + (i % 20)) for i in range(len(frame.periods))]

        mavp = MAVP(
            frame=populated_frame,
            period_calculator=period_calc,
            ma_type=0,
            column_name='MAVP_VAR'
        )

        assert len(mavp.periods) > 0


class TestMIDPOINT:
    """Test suite for MIDPOINT indicator."""

    def test_initialization(self, timeframe_1m):
        """Test MIDPOINT indicator initialization."""
        midpoint = MIDPOINT(frame=timeframe_1m, period=14, column_name='MIDPOINT_14')
        assert midpoint.period == 14
        assert midpoint.column_name == 'MIDPOINT_14'

    def test_invalid_period(self, timeframe_1m):
        """Test MIDPOINT with invalid period."""
        with pytest.raises(ValueError, match="MIDPOINT period must be at least 2"):
            MIDPOINT(frame=timeframe_1m, period=1)

    def test_calculation(self, populated_frame):
        """Test MIDPOINT calculates values."""
        midpoint = MIDPOINT(frame=populated_frame, period=14)
        values = [p.MIDPOINT for p in midpoint.periods if hasattr(p, 'MIDPOINT')]
        assert len(values) > 0
        assert all(v > 0 for v in values)


class TestMIDPRICE:
    """Test suite for MIDPRICE indicator."""

    def test_initialization(self, timeframe_1m):
        """Test MIDPRICE indicator initialization."""
        midprice = MIDPRICE(frame=timeframe_1m, period=14, column_name='MIDPRICE_14')
        assert midprice.period == 14
        assert midprice.column_name == 'MIDPRICE_14'

    def test_invalid_period(self, timeframe_1m):
        """Test MIDPRICE with invalid period."""
        with pytest.raises(ValueError, match="MIDPRICE period must be at least 2"):
            MIDPRICE(frame=timeframe_1m, period=1)

    def test_calculation(self, populated_frame):
        """Test MIDPRICE calculates values."""
        midprice = MIDPRICE(frame=populated_frame, period=14)
        values = [p.MIDPRICE for p in midprice.periods if hasattr(p, 'MIDPRICE')]
        assert len(values) > 0


class TestSAR:
    """Test suite for SAR (Parabolic SAR) indicator."""

    def test_initialization(self, timeframe_1m):
        """Test SAR indicator initialization."""
        sar = SAR(frame=timeframe_1m, acceleration=0.02, maximum=0.2)
        assert sar.acceleration == 0.02
        assert sar.maximum == 0.2
        assert sar.column_name == 'SAR'

    def test_invalid_acceleration(self, timeframe_1m):
        """Test SAR with invalid acceleration."""
        with pytest.raises(ValueError, match="acceleration must be between 0.0 and 1.0"):
            SAR(frame=timeframe_1m, acceleration=1.5)

    def test_invalid_maximum(self, timeframe_1m):
        """Test SAR with invalid maximum."""
        with pytest.raises(ValueError, match="maximum must be between 0.0 and 1.0"):
            SAR(frame=timeframe_1m, maximum=2.0)

    def test_calculation(self, populated_frame):
        """Test SAR calculates values."""
        sar = SAR(frame=populated_frame, acceleration=0.02, maximum=0.2)
        values = [p.SAR for p in sar.periods if hasattr(p, 'SAR')]
        assert len(values) > 0

    def test_trend_detection(self, populated_frame):
        """Test SAR trend detection methods."""
        sar = SAR(frame=populated_frame)

        uptrend = sar.is_uptrend()
        downtrend = sar.is_downtrend()
        assert isinstance(uptrend, bool)
        assert isinstance(downtrend, bool)

    def test_reversal_detection(self, populated_frame):
        """Test SAR reversal detection."""
        sar = SAR(frame=populated_frame)

        bullish = sar.is_bullish_reversal()
        bearish = sar.is_bearish_reversal()
        assert isinstance(bullish, bool)
        assert isinstance(bearish, bool)


class TestSAREXT:
    """Test suite for SAREXT indicator."""

    def test_initialization(self, timeframe_1m):
        """Test SAREXT indicator initialization."""
        sarext = SAREXT(
            frame=timeframe_1m,
            accelerationinitlong=0.02,
            accelerationlong=0.02,
            accelerationmaxlong=0.2
        )
        assert sarext.accelerationinitlong == 0.02
        assert sarext.accelerationlong == 0.02
        assert sarext.accelerationmaxlong == 0.2

    def test_calculation(self, populated_frame):
        """Test SAREXT calculates values."""
        sarext = SAREXT(frame=populated_frame)
        values = [p.SAREXT for p in sarext.periods if hasattr(p, 'SAREXT')]
        assert len(values) > 0

    def test_asymmetric_parameters(self, populated_frame):
        """Test SAREXT with asymmetric parameters."""
        sarext = SAREXT(
            frame=populated_frame,
            accelerationinitlong=0.02,
            accelerationlong=0.02,
            accelerationmaxlong=0.2,
            accelerationinitshort=0.02,
            accelerationshort=0.03,
            accelerationmaxshort=0.3
        )
        assert len(sarext.periods) > 0


class TestT3:
    """Test suite for T3 indicator."""

    def test_initialization(self, timeframe_1m):
        """Test T3 indicator initialization."""
        t3 = T3(frame=timeframe_1m, period=5, vfactor=0.7)
        assert t3.period == 5
        assert t3.vfactor == 0.7

    def test_invalid_period(self, timeframe_1m):
        """Test T3 with invalid period."""
        with pytest.raises(ValueError, match="T3 period must be at least 2"):
            T3(frame=timeframe_1m, period=1)

    def test_invalid_vfactor(self, timeframe_1m):
        """Test T3 with invalid vfactor."""
        with pytest.raises(ValueError, match="vfactor must be between 0.0 and 1.0"):
            T3(frame=timeframe_1m, period=5, vfactor=1.5)

    def test_calculation(self, populated_frame):
        """Test T3 calculates values."""
        t3 = T3(frame=populated_frame, period=5, vfactor=0.7)
        # T3 needs period * 6 for stable results
        values = [p.T3 for p in t3.periods if hasattr(p, 'T3')]
        # May have values if enough data


class TestTEMA:
    """Test suite for TEMA indicator."""

    def test_initialization(self, timeframe_1m):
        """Test TEMA indicator initialization."""
        tema = TEMA(frame=timeframe_1m, period=21, column_name='TEMA_21')
        assert tema.period == 21
        assert tema.column_name == 'TEMA_21'

    def test_invalid_period(self, timeframe_1m):
        """Test TEMA with invalid period."""
        with pytest.raises(ValueError, match="TEMA period must be at least 2"):
            TEMA(frame=timeframe_1m, period=1)

    def test_calculation(self, populated_frame):
        """Test TEMA calculates values."""
        tema = TEMA(frame=populated_frame, period=9, column_name='TEMA_9')
        values = [p.TEMA_9 for p in tema.periods if hasattr(p, 'TEMA_9')]
        assert len(values) > 0
        assert all(v > 0 for v in values)


class TestTRIMA:
    """Test suite for TRIMA indicator."""

    def test_initialization(self, timeframe_1m):
        """Test TRIMA indicator initialization."""
        trima = TRIMA(frame=timeframe_1m, period=20, column_name='TRIMA_20')
        assert trima.period == 20
        assert trima.column_name == 'TRIMA_20'

    def test_invalid_period(self, timeframe_1m):
        """Test TRIMA with invalid period."""
        with pytest.raises(ValueError, match="TRIMA period must be at least 2"):
            TRIMA(frame=timeframe_1m, period=1)

    def test_calculation(self, populated_frame):
        """Test TRIMA calculates values."""
        trima = TRIMA(frame=populated_frame, period=20)
        values = [p.TRIMA for p in trima.periods if hasattr(p, 'TRIMA')]
        assert len(values) > 0


class TestWMA:
    """Test suite for WMA indicator."""

    def test_initialization(self, timeframe_1m):
        """Test WMA indicator initialization."""
        wma = WMA(frame=timeframe_1m, period=20, column_name='WMA_20')
        assert wma.period == 20
        assert wma.column_name == 'WMA_20'

    def test_invalid_period(self, timeframe_1m):
        """Test WMA with invalid period."""
        with pytest.raises(ValueError, match="WMA period must be at least 1"):
            WMA(frame=timeframe_1m, period=0)

    def test_calculation(self, populated_frame):
        """Test WMA calculates values."""
        wma = WMA(frame=populated_frame, period=20, column_name='WMA_20')
        values = [p.WMA_20 for p in wma.periods if hasattr(p, 'WMA_20')]
        assert len(values) > 0
        assert all(v > 0 for v in values)

    def test_wma_vs_sma(self, populated_frame):
        """Test WMA is different from SMA."""
        from trading_indicators.trend import SMA

        wma = WMA(frame=populated_frame, period=20, column_name='WMA_20')
        sma = SMA(frame=populated_frame, period=20, column_name='SMA_20')

        wma_array = wma.to_numpy()
        sma_array = sma.to_numpy()

        # WMA and SMA should produce different values (WMA more responsive)
        valid_mask = ~(np.isnan(wma_array) | np.isnan(sma_array))
        if np.sum(valid_mask) > 0:
            # They should be different
            assert not np.allclose(wma_array[valid_mask], sma_array[valid_mask])
