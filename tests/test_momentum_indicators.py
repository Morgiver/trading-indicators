"""Tests for momentum indicators."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.trading_indicators.momentum import (
    ADX, ADXR, APO, AROON, AROONOSC, BOP, CCI, CMO, DX,
    MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM,
    PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI,
    STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR
)


@pytest.fixture
def mock_frame():
    """Create a mock frame with sample OHLCV data."""
    frame = Mock()
    frame.max_periods = 100

    # Create realistic price data
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
        volume = 1000 + np.random.randint(-100, 100)

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


class TestADX:
    """Tests for ADX indicator."""

    def test_initialization(self, mock_frame):
        """Test ADX initialization."""
        adx = ADX(mock_frame, length=14, column_name='ADX_14')
        assert adx.length == 14
        assert adx.column_name == 'ADX_14'
        assert len(adx.periods) > 0

    def test_calculation(self, mock_frame):
        """Test ADX calculation produces values."""
        adx = ADX(mock_frame, length=14)
        assert len(adx.periods) > 0

        # Check that some periods have ADX values
        values_count = sum(1 for p in adx.periods if hasattr(p, 'ADX'))
        assert values_count > 0

    def test_get_latest(self, mock_frame):
        """Test getting latest ADX value."""
        adx = ADX(mock_frame, length=14)
        latest = adx.get_latest()
        assert latest is None or isinstance(latest, float)

    def test_to_numpy(self, mock_frame):
        """Test numpy export."""
        adx = ADX(mock_frame, length=14)
        array = adx.to_numpy()
        assert isinstance(array, np.ndarray)
        assert len(array) == len(adx.periods)


class TestMACD:
    """Tests for MACD indicator."""

    def test_initialization(self, mock_frame):
        """Test MACD initialization."""
        macd = MACD(mock_frame, fast=12, slow=26, signal=9)
        assert macd.fast == 12
        assert macd.slow == 26
        assert macd.signal == 9
        assert len(macd.column_names) == 3

    def test_calculation(self, mock_frame):
        """Test MACD calculation produces values."""
        macd = MACD(mock_frame)
        assert len(macd.periods) > 0

        # Check that some periods have MACD values
        values_count = sum(1 for p in macd.periods if hasattr(p, 'MACD_LINE'))
        assert values_count > 0

    def test_to_numpy(self, mock_frame):
        """Test numpy export returns dictionary."""
        macd = MACD(mock_frame)
        arrays = macd.to_numpy()
        assert isinstance(arrays, dict)
        assert 'MACD_LINE' in arrays
        assert 'MACD_SIGNAL' in arrays
        assert 'MACD_HIST' in arrays


class TestRSI:
    """Tests for RSI indicator."""

    def test_initialization(self, mock_frame):
        """Test RSI initialization."""
        rsi = RSI(mock_frame, length=14, column_name='RSI_14')
        assert rsi.length == 14
        assert rsi.column_name == 'RSI_14'

    def test_calculation(self, mock_frame):
        """Test RSI calculation produces values between 0 and 100."""
        rsi = RSI(mock_frame, length=14)

        # Get values
        values = [getattr(p, 'RSI', None) for p in rsi.periods]
        valid_values = [v for v in values if v is not None]

        assert len(valid_values) > 0
        assert all(0 <= v <= 100 for v in valid_values)

    def test_overbought_oversold(self, mock_frame):
        """Test overbought/oversold detection."""
        rsi = RSI(mock_frame, length=14)

        # These should return boolean
        assert isinstance(rsi.is_overbought(), bool)
        assert isinstance(rsi.is_oversold(), bool)


class TestSTOCH:
    """Tests for STOCH indicator."""

    def test_initialization(self, mock_frame):
        """Test STOCH initialization."""
        stoch = STOCH(mock_frame, fastk_period=5, slowk_period=3, slowd_period=3)
        assert stoch.fastk_period == 5
        assert stoch.slowk_period == 3
        assert stoch.slowd_period == 3
        assert len(stoch.column_names) == 2

    def test_calculation(self, mock_frame):
        """Test STOCH calculation produces values."""
        stoch = STOCH(mock_frame)
        assert len(stoch.periods) > 0

        # Check that some periods have values
        values_count = sum(1 for p in stoch.periods if hasattr(p, 'STOCH_K'))
        assert values_count > 0

    def test_crossover_detection(self, mock_frame):
        """Test crossover detection methods."""
        stoch = STOCH(mock_frame)

        # These should return boolean
        assert isinstance(stoch.is_bullish_crossover(), bool)
        assert isinstance(stoch.is_bearish_crossover(), bool)


class TestCCI:
    """Tests for CCI indicator."""

    def test_initialization(self, mock_frame):
        """Test CCI initialization."""
        cci = CCI(mock_frame, length=14)
        assert cci.length == 14
        assert cci.column_name == 'CCI'

    def test_calculation(self, mock_frame):
        """Test CCI calculation produces values."""
        cci = CCI(mock_frame, length=14)
        values_count = sum(1 for p in cci.periods if hasattr(p, 'CCI'))
        assert values_count > 0


class TestMOM:
    """Tests for MOM indicator."""

    def test_initialization(self, mock_frame):
        """Test MOM initialization."""
        mom = MOM(mock_frame, length=10)
        assert mom.length == 10

    def test_bullish_bearish(self, mock_frame):
        """Test bullish/bearish detection."""
        mom = MOM(mock_frame, length=10)
        assert isinstance(mom.is_bullish(), bool)
        assert isinstance(mom.is_bearish(), bool)


class TestROCFamily:
    """Tests for ROC family of indicators."""

    def test_roc_initialization(self, mock_frame):
        """Test ROC initialization."""
        roc = ROC(mock_frame, length=10)
        assert roc.length == 10

    def test_rocp_initialization(self, mock_frame):
        """Test ROCP initialization."""
        rocp = ROCP(mock_frame, length=10)
        assert rocp.length == 10

    def test_rocr_initialization(self, mock_frame):
        """Test ROCR initialization."""
        rocr = ROCR(mock_frame, length=10)
        assert rocr.length == 10

    def test_rocr100_initialization(self, mock_frame):
        """Test ROCR100 initialization."""
        rocr100 = ROCR100(mock_frame, length=10)
        assert rocr100.length == 10


class TestAROON:
    """Tests for AROON and AROONOSC indicators."""

    def test_aroon_initialization(self, mock_frame):
        """Test AROON initialization."""
        aroon = AROON(mock_frame, length=14)
        assert aroon.length == 14
        assert len(aroon.column_names) == 2

    def test_aroon_calculation(self, mock_frame):
        """Test AROON produces two values."""
        aroon = AROON(mock_frame, length=14)
        arrays = aroon.to_numpy()
        assert isinstance(arrays, dict)
        assert len(arrays) == 2

    def test_aroonosc_initialization(self, mock_frame):
        """Test AROONOSC initialization."""
        aroonosc = AROONOSC(mock_frame, length=14)
        assert aroonosc.length == 14


class TestDirectionalIndicators:
    """Tests for directional movement indicators."""

    def test_plus_di(self, mock_frame):
        """Test PLUS_DI indicator."""
        plus_di = PLUS_DI(mock_frame, length=14)
        assert plus_di.length == 14

    def test_minus_di(self, mock_frame):
        """Test MINUS_DI indicator."""
        minus_di = MINUS_DI(mock_frame, length=14)
        assert minus_di.length == 14

    def test_plus_dm(self, mock_frame):
        """Test PLUS_DM indicator."""
        plus_dm = PLUS_DM(mock_frame, length=14)
        assert plus_dm.length == 14

    def test_minus_dm(self, mock_frame):
        """Test MINUS_DM indicator."""
        minus_dm = MINUS_DM(mock_frame, length=14)
        assert minus_dm.length == 14

    def test_dx(self, mock_frame):
        """Test DX indicator."""
        dx = DX(mock_frame, length=14)
        assert dx.length == 14


class TestMFI:
    """Tests for MFI indicator."""

    def test_initialization(self, mock_frame):
        """Test MFI initialization."""
        mfi = MFI(mock_frame, length=14)
        assert mfi.length == 14

    def test_calculation_with_volume(self, mock_frame):
        """Test MFI calculation (requires volume)."""
        mfi = MFI(mock_frame, length=14)
        values_count = sum(1 for p in mfi.periods if hasattr(p, 'MFI'))
        assert values_count > 0


class TestOscillators:
    """Tests for various oscillator indicators."""

    def test_bop(self, mock_frame):
        """Test BOP indicator."""
        bop = BOP(mock_frame)
        assert bop.column_name == 'BOP'

    def test_cmo(self, mock_frame):
        """Test CMO indicator."""
        cmo = CMO(mock_frame, length=14)
        assert cmo.length == 14

    def test_apo(self, mock_frame):
        """Test APO indicator."""
        apo = APO(mock_frame, fast=12, slow=26)
        assert apo.fast == 12
        assert apo.slow == 26

    def test_ppo(self, mock_frame):
        """Test PPO indicator."""
        ppo = PPO(mock_frame, fast=12, slow=26)
        assert ppo.fast == 12
        assert ppo.slow == 26


class TestAdvancedIndicators:
    """Tests for advanced indicators."""

    def test_trix(self, mock_frame):
        """Test TRIX indicator."""
        trix = TRIX(mock_frame, length=30)
        assert trix.length == 30

    def test_ultosc(self, mock_frame):
        """Test ULTOSC indicator."""
        ultosc = ULTOSC(mock_frame, period1=7, period2=14, period3=28)
        assert ultosc.period1 == 7
        assert ultosc.period2 == 14
        assert ultosc.period3 == 28

    def test_willr(self, mock_frame):
        """Test WILLR indicator."""
        willr = WILLR(mock_frame, length=14)
        assert willr.length == 14

    def test_stochf(self, mock_frame):
        """Test STOCHF indicator."""
        stochf = STOCHF(mock_frame, fastk_period=5, fastd_period=3)
        assert stochf.fastk_period == 5
        assert stochf.fastd_period == 3

    def test_stochrsi(self, mock_frame):
        """Test STOCHRSI indicator."""
        stochrsi = STOCHRSI(mock_frame, length=14, fastk_period=5, fastd_period=3)
        assert stochrsi.length == 14


class TestMACDVariants:
    """Tests for MACD variants."""

    def test_macdext(self, mock_frame):
        """Test MACDEXT indicator."""
        macdext = MACDEXT(mock_frame, fast=12, slow=26, signal=9)
        assert macdext.fast == 12
        assert macdext.slow == 26
        assert macdext.signal == 9
        assert len(macdext.column_names) == 3

    def test_macdfix(self, mock_frame):
        """Test MACDFIX indicator."""
        macdfix = MACDFIX(mock_frame, signal=9)
        assert macdfix.signal == 9
        assert len(macdfix.column_names) == 3


class TestADXRVariant:
    """Tests for ADXR indicator."""

    def test_initialization(self, mock_frame):
        """Test ADXR initialization."""
        adxr = ADXR(mock_frame, length=14)
        assert adxr.length == 14

    def test_calculation(self, mock_frame):
        """Test ADXR calculation produces values."""
        adxr = ADXR(mock_frame, length=14)
        values_count = sum(1 for p in adxr.periods if hasattr(p, 'ADXR'))
        assert values_count > 0


def test_all_indicators_export():
    """Test that all indicators are exported."""
    from src.trading_indicators import momentum

    expected_indicators = [
        'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
        'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM',
        'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI',
        'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 'ULTOSC', 'WILLR'
    ]

    assert set(momentum.__all__) == set(expected_indicators)
    assert len(momentum.__all__) == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
