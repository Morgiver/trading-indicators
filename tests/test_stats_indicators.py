"""Tests for statistical indicators."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from trading_indicators.stats import (
    BETA,
    CORREL,
    LINEARREG,
    LINEARREG_ANGLE,
    LINEARREG_INTERCEPT,
    LINEARREG_SLOPE,
    STDDEV,
    TSF,
    VAR,
)


@pytest.fixture
def mock_frame():
    """Create a mock frame with realistic price data."""
    frame = Mock()

    # Generate 100 periods with trending price data
    base_price = 100.0
    periods = []

    for i in range(100):
        period = Mock()
        period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)

        # Create uptrend with noise
        trend = i * 0.5  # Upward trend
        noise = np.random.normal(0, 2)
        close_price = base_price + trend + noise

        period.open_price = close_price + np.random.normal(0, 1)
        period.close_price = close_price
        period.high_price = close_price + abs(np.random.normal(1, 0.5))
        period.low_price = close_price - abs(np.random.normal(1, 0.5))
        period.volume = 1000000 + np.random.randint(-100000, 100000)

        periods.append(period)

    frame.periods = periods
    frame.max_periods = 100

    return frame


class TestStaticUtilityMode:
    """Test static compute() methods (utility mode)."""

    def test_stddev_compute(self):
        """Test STDDEV static compute method."""
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], dtype=np.float64)
        result = STDDEV.compute(prices, length=5, nbdev=1)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices)
        assert not np.isnan(result[-1])

    def test_var_compute(self):
        """Test VAR static compute method."""
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], dtype=np.float64)
        result = VAR.compute(prices, length=5, nbdev=1)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])

    def test_linearreg_compute(self):
        """Test LINEARREG static compute method."""
        prices = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        result = LINEARREG.compute(prices, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])

    def test_linearreg_angle_compute(self):
        """Test LINEARREG_ANGLE static compute method."""
        prices = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        result = LINEARREG_ANGLE.compute(prices, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])

    def test_linearreg_slope_compute(self):
        """Test LINEARREG_SLOPE static compute method."""
        prices = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        result = LINEARREG_SLOPE.compute(prices, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])
        assert result[-1] > 0  # Uptrending

    def test_linearreg_intercept_compute(self):
        """Test LINEARREG_INTERCEPT static compute method."""
        prices = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        result = LINEARREG_INTERCEPT.compute(prices, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])

    def test_tsf_compute(self):
        """Test TSF static compute method."""
        prices = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        result = TSF.compute(prices, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])
        # Forecast should be higher than last price for uptrend
        assert result[-1] > prices[-1]

    def test_correl_compute(self):
        """Test CORREL static compute method."""
        series1 = np.array([100, 102, 104, 106, 108], dtype=np.float64)
        series2 = np.array([50, 51, 52, 53, 54], dtype=np.float64)
        result = CORREL.compute(series1, series2, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])
        # Perfect positive correlation
        assert result[-1] > 0.9

    def test_beta_compute(self):
        """Test BETA static compute method."""
        # BETA requires length + 1 periods of data (5 + 1 = 6)
        series1 = np.array([100, 102, 104, 106, 108, 110], dtype=np.float64)
        series2 = np.array([50, 51, 52, 53, 54, 55], dtype=np.float64)
        result = BETA.compute(series1, series2, length=5)

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result[-1])


class TestSTDDEV:
    """Tests for STDDEV (Standard Deviation) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = STDDEV(frame=mock_frame, length=20, column_name='STDDEV')

        assert indicator.length == 20
        assert indicator.column_name == 'STDDEV'
        assert indicator.frame == mock_frame

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = STDDEV(frame=mock_frame, length=20)

        # Calculate for all periods
        for period in mock_frame.periods:
            indicator.calculate(period)

        # Should have values
        latest = indicator.get_latest()
        assert latest is not None
        assert latest > 0

    def test_is_high_volatility(self, mock_frame):
        """Test high volatility detection."""
        indicator = STDDEV(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_high_volatility(threshold=1.0)
        assert isinstance(result, bool)

    def test_is_low_volatility(self, mock_frame):
        """Test low volatility detection."""
        indicator = STDDEV(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_low_volatility(threshold=10.0)
        assert isinstance(result, bool)

    def test_is_expanding(self, mock_frame):
        """Test volatility expansion detection."""
        indicator = STDDEV(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_expanding()
        assert isinstance(result, bool)

    def test_is_contracting(self, mock_frame):
        """Test volatility contraction detection."""
        indicator = STDDEV(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_contracting()
        assert isinstance(result, bool)


class TestVAR:
    """Tests for VAR (Variance) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = VAR(frame=mock_frame, length=20, column_name='VAR')

        assert indicator.length == 20
        assert indicator.column_name == 'VAR'

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = VAR(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None
        assert latest >= 0

    def test_is_high_variance(self, mock_frame):
        """Test high variance detection."""
        indicator = VAR(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_high_variance(threshold=1.0)
        assert isinstance(result, bool)

    def test_is_low_variance(self, mock_frame):
        """Test low variance detection."""
        indicator = VAR(frame=mock_frame, length=20)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_low_variance(threshold=100.0)
        assert isinstance(result, bool)


class TestLINEARREG:
    """Tests for LINEARREG (Linear Regression) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = LINEARREG(frame=mock_frame, length=14, column_name='LINEARREG')

        assert indicator.length == 14
        assert indicator.column_name == 'LINEARREG'

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = LINEARREG(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None

    def test_is_above_regression(self, mock_frame):
        """Test price above regression detection."""
        indicator = LINEARREG(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_above_regression()
        assert isinstance(result, bool)

    def test_is_below_regression(self, mock_frame):
        """Test price below regression detection."""
        indicator = LINEARREG(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_below_regression()
        assert isinstance(result, bool)

    def test_get_distance_from_regression(self, mock_frame):
        """Test distance from regression calculation."""
        indicator = LINEARREG(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        distance = indicator.get_distance_from_regression()
        assert distance is not None
        assert isinstance(distance, float)


class TestLINEARREG_ANGLE:
    """Tests for LINEARREG_ANGLE indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = LINEARREG_ANGLE(frame=mock_frame, length=14, column_name='REG_ANGLE')

        assert indicator.length == 14
        assert indicator.column_name == 'REG_ANGLE'

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = LINEARREG_ANGLE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None

    def test_is_steep_uptrend(self, mock_frame):
        """Test steep uptrend detection."""
        indicator = LINEARREG_ANGLE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_steep_uptrend(threshold=30)
        assert isinstance(result, bool)

    def test_is_steep_downtrend(self, mock_frame):
        """Test steep downtrend detection."""
        indicator = LINEARREG_ANGLE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_steep_downtrend(threshold=-30)
        assert isinstance(result, bool)

    def test_is_flat(self, mock_frame):
        """Test flat market detection."""
        indicator = LINEARREG_ANGLE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_flat(threshold=10)
        assert isinstance(result, bool)


class TestLINEARREG_SLOPE:
    """Tests for LINEARREG_SLOPE indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = LINEARREG_SLOPE(frame=mock_frame, length=14, column_name='REG_SLOPE')

        assert indicator.length == 14
        assert indicator.column_name == 'REG_SLOPE'

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = LINEARREG_SLOPE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None

    def test_is_positive_slope(self, mock_frame):
        """Test positive slope detection."""
        indicator = LINEARREG_SLOPE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_positive_slope()
        assert isinstance(result, bool)

    def test_is_negative_slope(self, mock_frame):
        """Test negative slope detection."""
        indicator = LINEARREG_SLOPE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_negative_slope()
        assert isinstance(result, bool)

    def test_is_accelerating(self, mock_frame):
        """Test acceleration detection."""
        indicator = LINEARREG_SLOPE(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_accelerating()
        assert isinstance(result, bool)


class TestLINEARREG_INTERCEPT:
    """Tests for LINEARREG_INTERCEPT indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = LINEARREG_INTERCEPT(frame=mock_frame, length=14)

        assert indicator.length == 14

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = LINEARREG_INTERCEPT(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None


class TestTSF:
    """Tests for TSF (Time Series Forecast) indicator."""

    def test_initialization(self, mock_frame):
        """Test indicator initialization."""
        indicator = TSF(frame=mock_frame, length=14, column_name='TSF')

        assert indicator.length == 14
        assert indicator.column_name == 'TSF'

    def test_calculate(self, mock_frame):
        """Test calculation."""
        indicator = TSF(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        latest = indicator.get_latest()
        assert latest is not None

    def test_is_price_above_forecast(self, mock_frame):
        """Test price above forecast detection."""
        indicator = TSF(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_price_above_forecast()
        assert isinstance(result, bool)

    def test_is_price_below_forecast(self, mock_frame):
        """Test price below forecast detection."""
        indicator = TSF(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        result = indicator.is_price_below_forecast()
        assert isinstance(result, bool)

    def test_get_forecast_error(self, mock_frame):
        """Test forecast error calculation."""
        indicator = TSF(frame=mock_frame, length=14)

        for period in mock_frame.periods:
            indicator.calculate(period)

        error = indicator.get_forecast_error()
        assert error is not None
        assert isinstance(error, float)


class TestCORREL:
    """Tests for CORREL (Correlation) indicator."""

    def test_initialization_utility_mode(self):
        """Test indicator in utility mode (no frame)."""
        indicator = CORREL(frame=None, length=30)

        assert indicator.frame is None
        assert indicator.length == 30

    def test_static_compute_positive_correlation(self):
        """Test perfect positive correlation."""
        series1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        series2 = np.array([2, 4, 6, 8, 10], dtype=np.float64)

        result = CORREL.compute(series1, series2, length=5)
        # Should be close to 1.0 (perfect positive correlation)
        assert result[-1] > 0.99

    def test_static_compute_negative_correlation(self):
        """Test negative correlation."""
        series1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        series2 = np.array([10, 8, 6, 4, 2], dtype=np.float64)

        result = CORREL.compute(series1, series2, length=5)
        # Should be close to -1.0 (perfect negative correlation)
        assert result[-1] < -0.99


class TestBETA:
    """Tests for BETA (Beta Coefficient) indicator."""

    def test_initialization_utility_mode(self):
        """Test indicator in utility mode (no frame)."""
        indicator = BETA(frame=None, length=5)

        assert indicator.frame is None
        assert indicator.length == 5

    def test_static_compute(self):
        """Test beta computation."""
        # BETA requires length + 1 periods of data (5 + 1 = 6)
        # Using realistic price data with correlation
        benchmark = np.array([100, 102, 104, 106, 108, 110], dtype=np.float64)
        asset = np.array([100, 104, 108, 112, 116, 120], dtype=np.float64)

        result = BETA.compute(asset, benchmark, length=5)
        # Beta should be computed (not NaN)
        assert not np.isnan(result[-1])
        # Beta should be a reasonable value
        assert isinstance(result[-1], (float, np.floating))
        assert result[-1] > 0  # Positive correlation
