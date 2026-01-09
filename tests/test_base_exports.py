"""Tests for BaseIndicator export methods (to_pandas, to_normalized)."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.trading_indicators.momentum import RSI, MACD
from src.trading_indicators.stats import STDDEV


@pytest.fixture
def mock_frame():
    """Create a mock frame with sample OHLC data."""
    frame = Mock()
    frame.max_periods = 100

    # Create realistic price data
    base_price = 100.0
    periods = []

    for i in range(50):
        # Simulate price movement
        close = base_price + i * 0.5 + np.random.randn() * 2
        high = close + abs(np.random.randn())
        low = close - abs(np.random.randn())
        open_price = close + np.random.randn() * 0.5

        period = Mock()
        period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)
        period.close_date = None
        period.open_price = max(low, min(high, open_price))
        period.high_price = high
        period.low_price = low
        period.close_price = close
        periods.append(period)

    frame.periods = periods
    frame.on = MagicMock()

    return frame


class TestToPandas:
    """Tests for to_pandas() export method."""

    def test_single_value_indicator_returns_series(self, mock_frame):
        """Test that single-value indicators return pandas Series."""
        rsi = RSI(mock_frame, length=14)
        result = rsi.to_pandas()

        assert isinstance(result, pd.Series)
        assert result.name == 'RSI'  # Column name from RSI indicator
        assert len(result) == len(rsi.periods)

    def test_multi_value_indicator_returns_dataframe(self, mock_frame):
        """Test that multi-value indicators return pandas DataFrame."""
        macd = MACD(mock_frame, fast=12, slow=26, signal=9)
        result = macd.to_pandas()

        assert isinstance(result, pd.DataFrame)
        assert 'MACD_LINE' in result.columns
        assert 'MACD_SIGNAL' in result.columns
        assert 'MACD_HIST' in result.columns
        assert len(result) == len(macd.periods)

    def test_pandas_index_is_timestamps(self, mock_frame):
        """Test that pandas index contains timestamps from periods."""
        rsi = RSI(mock_frame, length=14)
        result = rsi.to_pandas()

        # Check that index matches period timestamps
        expected_timestamps = [p.open_date for p in rsi.periods]
        assert list(result.index) == expected_timestamps

    def test_pandas_values_match_numpy(self, mock_frame):
        """Test that pandas values match numpy export."""
        rsi = RSI(mock_frame, length=14)
        pandas_result = rsi.to_pandas()
        numpy_result = rsi.to_numpy()

        # Compare values (handling NaN)
        np.testing.assert_allclose(
            pandas_result.values,
            numpy_result,
            equal_nan=True
        )

    def test_pandas_empty_indicator(self, mock_frame):
        """Test to_pandas on indicator with no periods."""
        frame = Mock()
        frame.max_periods = 100
        frame.periods = []
        frame.on = MagicMock()

        rsi = RSI(frame, length=14)
        result = rsi.to_pandas()

        assert isinstance(result, pd.Series)
        assert len(result) == 0


class TestToNormalized:
    """Tests for to_normalized() export method."""

    def test_minmax_normalization_default_range(self, mock_frame):
        """Test min-max normalization with default range (0, 1)."""
        rsi = RSI(mock_frame, length=14)
        normalized = rsi.to_normalized(method='minmax')

        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(rsi.periods)

        # Check range (ignoring NaN values)
        valid_values = normalized[~np.isnan(normalized)]
        if len(valid_values) > 0:
            assert np.min(valid_values) >= 0.0
            assert np.max(valid_values) <= 1.0

    def test_minmax_normalization_custom_range(self, mock_frame):
        """Test min-max normalization with custom range (-1, 1)."""
        rsi = RSI(mock_frame, length=14)
        normalized = rsi.to_normalized(method='minmax', feature_range=(-1, 1))

        assert isinstance(normalized, np.ndarray)

        # Check range (ignoring NaN values)
        valid_values = normalized[~np.isnan(normalized)]
        if len(valid_values) > 0:
            assert np.min(valid_values) >= -1.0
            assert np.max(valid_values) <= 1.0

    def test_zscore_normalization(self, mock_frame):
        """Test z-score normalization."""
        rsi = RSI(mock_frame, length=14)
        normalized = rsi.to_normalized(method='zscore')

        assert isinstance(normalized, np.ndarray)

        # Check that mean is close to 0 and std close to 1 (ignoring NaN)
        valid_values = normalized[~np.isnan(normalized)]
        if len(valid_values) > 5:  # Need enough values
            assert abs(np.mean(valid_values)) < 0.1
            assert abs(np.std(valid_values) - 1.0) < 0.1

    def test_normalization_preserves_nan(self, mock_frame):
        """Test that NaN values are preserved in normalization."""
        rsi = RSI(mock_frame, length=14)
        original = rsi.to_numpy()
        normalized = rsi.to_normalized(method='minmax')

        # Check that NaN positions are preserved
        original_nan_mask = np.isnan(original)
        normalized_nan_mask = np.isnan(normalized)
        np.testing.assert_array_equal(original_nan_mask, normalized_nan_mask)

    def test_normalization_multi_value_indicator(self, mock_frame):
        """Test normalization on multi-value indicator (dict output)."""
        macd = MACD(mock_frame, fast=12, slow=26, signal=9)
        normalized = macd.to_normalized(method='minmax')

        assert isinstance(normalized, dict)
        assert 'MACD_LINE' in normalized
        assert 'MACD_SIGNAL' in normalized
        assert 'MACD_HIST' in normalized

        # Check each array is normalized
        for key, arr in normalized.items():
            assert isinstance(arr, np.ndarray)
            valid_values = arr[~np.isnan(arr)]
            if len(valid_values) > 0:
                assert np.min(valid_values) >= 0.0
                assert np.max(valid_values) <= 1.0

    def test_normalization_invalid_method(self, mock_frame):
        """Test that invalid normalization method raises ValueError."""
        rsi = RSI(mock_frame, length=14)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            rsi.to_normalized(method='invalid_method')

    def test_normalization_constant_values(self, mock_frame):
        """Test normalization when all values are constant."""
        # Create frame with constant prices
        frame = Mock()
        frame.max_periods = 100
        periods = []

        for i in range(50):
            period = Mock()
            period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)
            period.close_date = None
            period.open_price = 100.0
            period.high_price = 100.0
            period.low_price = 100.0
            period.close_price = 100.0
            periods.append(period)

        frame.periods = periods
        frame.on = MagicMock()

        stddev = STDDEV(frame, length=10)
        normalized = stddev.to_normalized(method='minmax')

        # Should return all zeros or feature_range minimum
        assert isinstance(normalized, np.ndarray)

    def test_normalization_all_nan(self, mock_frame):
        """Test normalization when all values are NaN."""
        # Create frame with minimal data (will produce all NaN for large period)
        frame = Mock()
        frame.max_periods = 100
        periods = []

        for i in range(5):  # Very few periods
            period = Mock()
            period.open_date = datetime(2024, 1, 1) + timedelta(hours=i)
            period.close_date = None
            period.open_price = 100.0
            period.high_price = 100.0
            period.low_price = 100.0
            period.close_price = 100.0
            periods.append(period)

        frame.periods = periods
        frame.on = MagicMock()

        rsi = RSI(frame, length=50)  # Period too large for data
        normalized = rsi.to_normalized(method='minmax')

        # Should return all NaN
        assert np.all(np.isnan(normalized))


class TestExportMethodsIntegration:
    """Integration tests for export methods."""

    def test_chain_numpy_pandas_normalized(self, mock_frame):
        """Test that all export methods work together."""
        rsi = RSI(mock_frame, length=14)

        # Export to numpy
        numpy_result = rsi.to_numpy()
        assert isinstance(numpy_result, np.ndarray)

        # Export to pandas
        pandas_result = rsi.to_pandas()
        assert isinstance(pandas_result, pd.Series)

        # Export normalized
        normalized_result = rsi.to_normalized(method='minmax')
        assert isinstance(normalized_result, np.ndarray)

        # All should have same length
        assert len(numpy_result) == len(pandas_result) == len(normalized_result)

    def test_normalized_pandas_combination(self, mock_frame):
        """Test creating pandas Series from normalized values."""
        rsi = RSI(mock_frame, length=14)

        # Get normalized values and create pandas Series manually
        normalized = rsi.to_normalized(method='minmax', feature_range=(-1, 1))
        timestamps = [p.open_date for p in rsi.periods]
        series = pd.Series(normalized, index=timestamps, name='RSI_normalized')

        assert isinstance(series, pd.Series)
        assert len(series) == len(rsi.periods)
        assert series.name == 'RSI_normalized'

    def test_multi_indicator_dataframe_concatenation(self, mock_frame):
        """Test concatenating multiple indicators into single DataFrame."""
        rsi = RSI(mock_frame, length=14)
        stddev = STDDEV(mock_frame, length=10)

        # Export to pandas
        rsi_series = rsi.to_pandas()
        stddev_series = stddev.to_pandas()

        # Concatenate into DataFrame
        df = pd.concat([rsi_series, stddev_series], axis=1)

        assert isinstance(df, pd.DataFrame)
        assert 'RSI' in df.columns
        assert 'STDDEV' in df.columns
        assert len(df) == len(rsi.periods)
