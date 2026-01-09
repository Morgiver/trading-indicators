"""Trading Indicators - Technical indicators library with frame synchronization."""

from .base import IndicatorPeriod, BaseIndicator
from .momentum.rsi import RSI
from .momentum.macd import MACD
from .trend.sma import SMA
from .trend.ema import EMA
from .trend.bollinger import BollingerBands
from .trend.pivot_points import PivotPoints
from .trend.fvg import FVG
from .volatility.atr import ATR
from .registry import (
    INDICATOR_REGISTRY,
    get_indicator_class,
    create_indicator,
    list_available_indicators,
    register_indicator,
)

__version__ = "0.2.0"
__all__ = [
    "IndicatorPeriod",
    "BaseIndicator",
    # Momentum
    "RSI",
    "MACD",
    # Trend
    "SMA",
    "EMA",
    "BollingerBands",
    "PivotPoints",
    "FVG",
    # Volatility
    "ATR",
    # Registry API
    "INDICATOR_REGISTRY",
    "get_indicator_class",
    "create_indicator",
    "list_available_indicators",
    "register_indicator",
]
