"""Indicator registry for textual API access.

This file is auto-generated. Run scripts/generate_registry.py to update.
"""

from typing import Dict, Type, Union, List, Any
from .base import BaseIndicator

# Import all indicators
from .cycle.ht_dcperiod import HT_DCPERIOD
from .cycle.ht_dcphase import HT_DCPHASE
from .cycle.ht_phasor import HT_PHASOR
from .cycle.ht_sine import HT_SINE
from .cycle.ht_trendmode import HT_TRENDMODE
from .momentum.adx import ADX
from .momentum.adxr import ADXR
from .momentum.apo import APO
from .momentum.aroon import AROON
from .momentum.aroonosc import AROONOSC
from .momentum.bop import BOP
from .momentum.cci import CCI
from .momentum.cmo import CMO
from .momentum.dx import DX
from .momentum.macd import MACD
from .momentum.macdext import MACDEXT
from .momentum.macdfix import MACDFIX
from .momentum.mfi import MFI
from .momentum.minus_di import MINUS_DI
from .momentum.minus_dm import MINUS_DM
from .momentum.mom import MOM
from .momentum.plus_di import PLUS_DI
from .momentum.plus_dm import PLUS_DM
from .momentum.ppo import PPO
from .momentum.roc import ROC
from .momentum.rocp import ROCP
from .momentum.rocr import ROCR
from .momentum.rocr100 import ROCR100
from .momentum.rsi import RSI
from .momentum.stoch import STOCH
from .momentum.stochf import STOCHF
from .momentum.stochrsi import STOCHRSI
from .momentum.trix import TRIX
from .momentum.ultosc import ULTOSC
from .momentum.willr import WILLR
from .pattern_recognition.cdl2crows import CDL2CROWS
from .pattern_recognition.cdl3blackcrows import CDL3BLACKCROWS
from .pattern_recognition.cdl3inside import CDL3INSIDE
from .pattern_recognition.cdl3linestrike import CDL3LINESTRIKE
from .pattern_recognition.cdl3outside import CDL3OUTSIDE
from .pattern_recognition.cdl3starsinsouth import CDL3STARSINSOUTH
from .pattern_recognition.cdl3whitesoldiers import CDL3WHITESOLDIERS
from .pattern_recognition.cdlabandonedbaby import CDLABANDONEDBABY
from .pattern_recognition.cdladvanceblock import CDLADVANCEBLOCK
from .pattern_recognition.cdlbelthold import CDLBELTHOLD
from .pattern_recognition.cdlbreakaway import CDLBREAKAWAY
from .pattern_recognition.cdlclosingmarubozu import CDLCLOSINGMARUBOZU
from .pattern_recognition.cdlconcealbabyswall import CDLCONCEALBABYSWALL
from .pattern_recognition.cdlcounterattack import CDLCOUNTERATTACK
from .pattern_recognition.cdldarkcloudcover import CDLDARKCLOUDCOVER
from .pattern_recognition.cdldoji import CDLDOJI
from .pattern_recognition.cdldojistar import CDLDOJISTAR
from .pattern_recognition.cdldragonflydoji import CDLDRAGONFLYDOJI
from .pattern_recognition.cdlengulfing import CDLENGULFING
from .pattern_recognition.cdleveningdojistar import CDLEVENINGDOJISTAR
from .pattern_recognition.cdleveningstar import CDLEVENINGSTAR
from .pattern_recognition.cdlgapsidesidewhite import CDLGAPSIDESIDEWHITE
from .pattern_recognition.cdlgravestonedoji import CDLGRAVESTONEDOJI
from .pattern_recognition.cdlhammer import CDLHAMMER
from .pattern_recognition.cdlhangingman import CDLHANGINGMAN
from .pattern_recognition.cdlharami import CDLHARAMI
from .pattern_recognition.cdlharamicross import CDLHARAMICROSS
from .pattern_recognition.cdlhighwave import CDLHIGHWAVE
from .pattern_recognition.cdlhikkake import CDLHIKKAKE
from .pattern_recognition.cdlhikkakemod import CDLHIKKAKEMOD
from .pattern_recognition.cdlhomingpigeon import CDLHOMINGPIGEON
from .pattern_recognition.cdlidentical3crows import CDLIDENTICAL3CROWS
from .pattern_recognition.cdlinneck import CDLINNECK
from .pattern_recognition.cdlinvertedhammer import CDLINVERTEDHAMMER
from .pattern_recognition.cdlkicking import CDLKICKING
from .pattern_recognition.cdlkickingbylength import CDLKICKINGBYLENGTH
from .pattern_recognition.cdlladderbottom import CDLLADDERBOTTOM
from .pattern_recognition.cdllongleggeddoji import CDLLONGLEGGEDDOJI
from .pattern_recognition.cdllongline import CDLLONGLINE
from .pattern_recognition.cdlmarubozu import CDLMARUBOZU
from .pattern_recognition.cdlmatchinglow import CDLMATCHINGLOW
from .pattern_recognition.cdlmathold import CDLMATHOLD
from .pattern_recognition.cdlmorningdojistar import CDLMORNINGDOJISTAR
from .pattern_recognition.cdlmorningstar import CDLMORNINGSTAR
from .pattern_recognition.cdlonneck import CDLONNECK
from .pattern_recognition.cdlpiercing import CDLPIERCING
from .pattern_recognition.cdlrickshawman import CDLRICKSHAWMAN
from .pattern_recognition.cdlrisefall3methods import CDLRISEFALL3METHODS
from .pattern_recognition.cdlseparatinglines import CDLSEPARATINGLINES
from .pattern_recognition.cdlshootingstar import CDLSHOOTINGSTAR
from .pattern_recognition.cdlshortline import CDLSHORTLINE
from .pattern_recognition.cdlspinningtop import CDLSPINNINGTOP
from .pattern_recognition.cdlstalledpattern import CDLSTALLEDPATTERN
from .pattern_recognition.cdlsticksandwich import CDLSTICKSANDWICH
from .pattern_recognition.cdltakuri import CDLTAKURI
from .pattern_recognition.cdltasukigap import CDLTASUKIGAP
from .pattern_recognition.cdlthrusting import CDLTHRUSTING
from .pattern_recognition.cdltristar import CDLTRISTAR
from .pattern_recognition.cdlunique3river import CDLUNIQUE3RIVER
from .pattern_recognition.cdlupsidegap2crows import CDLUPSIDEGAP2CROWS
from .pattern_recognition.cdlxsidegap3methods import CDLXSIDEGAP3METHODS
from .price.avgprice import AVGPRICE
from .price.medprice import MEDPRICE
from .price.typprice import TYPPRICE
from .price.wclprice import WCLPRICE
from .stats.beta import BETA
from .stats.correl import CORREL
from .stats.linearreg import LINEARREG
from .stats.linearreg_angle import LINEARREG_ANGLE
from .stats.linearreg_intercept import LINEARREG_INTERCEPT
from .stats.linearreg_slope import LINEARREG_SLOPE
from .stats.stddev import STDDEV
from .stats.tsf import TSF
from .stats.var import VAR
from .trend.bollinger import BollingerBands
from .trend.dema import DEMA
from .trend.ema import EMA
from .trend.fvg import FVG
from .trend.ht_trendline import HT_TRENDLINE
from .trend.kama import KAMA
from .trend.ma import MA
from .trend.mama import MAMA
from .trend.mavp import MAVP
from .trend.midpoint import MIDPOINT
from .trend.midprice import MIDPRICE
from .trend.pivot_points import PivotPoints
from .trend.sar import SAR
from .trend.sarext import SAREXT
from .trend.sma import SMA
from .trend.t3 import T3
from .trend.tema import TEMA
from .trend.trima import TRIMA
from .trend.wma import WMA
from .volatility.atr import ATR
from .volume.ad import AD
from .volume.adosc import ADOSC
from .volume.obv import OBV


# Registry mapping string names to indicator classes
INDICATOR_REGISTRY: Dict[str, Type[BaseIndicator]] = {
    # Cycle indicators
    "HT_DCPERIOD": HT_DCPERIOD,
    "HT_DCPHASE": HT_DCPHASE,
    "HT_PHASOR": HT_PHASOR,
    "HT_SINE": HT_SINE,
    "HT_TRENDMODE": HT_TRENDMODE,

    # Momentum indicators
    "ADX": ADX,
    "ADXR": ADXR,
    "APO": APO,
    "AROON": AROON,
    "AROONOSC": AROONOSC,
    "BOP": BOP,
    "CCI": CCI,
    "CMO": CMO,
    "DX": DX,
    "MACD": MACD,
    "MACDEXT": MACDEXT,
    "MACDFIX": MACDFIX,
    "MFI": MFI,
    "MINUS_DI": MINUS_DI,
    "MINUS_DM": MINUS_DM,
    "MOM": MOM,
    "PLUS_DI": PLUS_DI,
    "PLUS_DM": PLUS_DM,
    "PPO": PPO,
    "ROC": ROC,
    "ROCP": ROCP,
    "ROCR": ROCR,
    "ROCR100": ROCR100,
    "RSI": RSI,
    "STOCH": STOCH,
    "STOCHF": STOCHF,
    "STOCHRSI": STOCHRSI,
    "TRIX": TRIX,
    "ULTOSC": ULTOSC,
    "WILLR": WILLR,

    # Pattern_recognition indicators
    "CDL2CROWS": CDL2CROWS,
    "CDL3BLACKCROWS": CDL3BLACKCROWS,
    "CDL3INSIDE": CDL3INSIDE,
    "CDL3LINESTRIKE": CDL3LINESTRIKE,
    "CDL3OUTSIDE": CDL3OUTSIDE,
    "CDL3STARSINSOUTH": CDL3STARSINSOUTH,
    "CDL3WHITESOLDIERS": CDL3WHITESOLDIERS,
    "CDLABANDONEDBABY": CDLABANDONEDBABY,
    "CDLADVANCEBLOCK": CDLADVANCEBLOCK,
    "CDLBELTHOLD": CDLBELTHOLD,
    "CDLBREAKAWAY": CDLBREAKAWAY,
    "CDLCLOSINGMARUBOZU": CDLCLOSINGMARUBOZU,
    "CDLCONCEALBABYSWALL": CDLCONCEALBABYSWALL,
    "CDLCOUNTERATTACK": CDLCOUNTERATTACK,
    "CDLDARKCLOUDCOVER": CDLDARKCLOUDCOVER,
    "CDLDOJI": CDLDOJI,
    "CDLDOJISTAR": CDLDOJISTAR,
    "CDLDRAGONFLYDOJI": CDLDRAGONFLYDOJI,
    "CDLENGULFING": CDLENGULFING,
    "CDLEVENINGDOJISTAR": CDLEVENINGDOJISTAR,
    "CDLEVENINGSTAR": CDLEVENINGSTAR,
    "CDLGAPSIDESIDEWHITE": CDLGAPSIDESIDEWHITE,
    "CDLGRAVESTONEDOJI": CDLGRAVESTONEDOJI,
    "CDLHAMMER": CDLHAMMER,
    "CDLHANGINGMAN": CDLHANGINGMAN,
    "CDLHARAMI": CDLHARAMI,
    "CDLHARAMICROSS": CDLHARAMICROSS,
    "CDLHIGHWAVE": CDLHIGHWAVE,
    "CDLHIKKAKE": CDLHIKKAKE,
    "CDLHIKKAKEMOD": CDLHIKKAKEMOD,
    "CDLHOMINGPIGEON": CDLHOMINGPIGEON,
    "CDLIDENTICAL3CROWS": CDLIDENTICAL3CROWS,
    "CDLINNECK": CDLINNECK,
    "CDLINVERTEDHAMMER": CDLINVERTEDHAMMER,
    "CDLKICKING": CDLKICKING,
    "CDLKICKINGBYLENGTH": CDLKICKINGBYLENGTH,
    "CDLLADDERBOTTOM": CDLLADDERBOTTOM,
    "CDLLONGLEGGEDDOJI": CDLLONGLEGGEDDOJI,
    "CDLLONGLINE": CDLLONGLINE,
    "CDLMARUBOZU": CDLMARUBOZU,
    "CDLMATCHINGLOW": CDLMATCHINGLOW,
    "CDLMATHOLD": CDLMATHOLD,
    "CDLMORNINGDOJISTAR": CDLMORNINGDOJISTAR,
    "CDLMORNINGSTAR": CDLMORNINGSTAR,
    "CDLONNECK": CDLONNECK,
    "CDLPIERCING": CDLPIERCING,
    "CDLRICKSHAWMAN": CDLRICKSHAWMAN,
    "CDLRISEFALL3METHODS": CDLRISEFALL3METHODS,
    "CDLSEPARATINGLINES": CDLSEPARATINGLINES,
    "CDLSHOOTINGSTAR": CDLSHOOTINGSTAR,
    "CDLSHORTLINE": CDLSHORTLINE,
    "CDLSPINNINGTOP": CDLSPINNINGTOP,
    "CDLSTALLEDPATTERN": CDLSTALLEDPATTERN,
    "CDLSTICKSANDWICH": CDLSTICKSANDWICH,
    "CDLTAKURI": CDLTAKURI,
    "CDLTASUKIGAP": CDLTASUKIGAP,
    "CDLTHRUSTING": CDLTHRUSTING,
    "CDLTRISTAR": CDLTRISTAR,
    "CDLUNIQUE3RIVER": CDLUNIQUE3RIVER,
    "CDLUPSIDEGAP2CROWS": CDLUPSIDEGAP2CROWS,
    "CDLXSIDEGAP3METHODS": CDLXSIDEGAP3METHODS,

    # Price indicators
    "AVGPRICE": AVGPRICE,
    "MEDPRICE": MEDPRICE,
    "TYPPRICE": TYPPRICE,
    "WCLPRICE": WCLPRICE,

    # Stats indicators
    "BETA": BETA,
    "CORREL": CORREL,
    "LINEARREG": LINEARREG,
    "LINEARREG_ANGLE": LINEARREG_ANGLE,
    "LINEARREG_INTERCEPT": LINEARREG_INTERCEPT,
    "LINEARREG_SLOPE": LINEARREG_SLOPE,
    "STDDEV": STDDEV,
    "TSF": TSF,
    "VAR": VAR,

    # Trend indicators
    "BollingerBands": BollingerBands,
    "DEMA": DEMA,
    "EMA": EMA,
    "FVG": FVG,
    "HT_TRENDLINE": HT_TRENDLINE,
    "KAMA": KAMA,
    "MA": MA,
    "MAMA": MAMA,
    "MAVP": MAVP,
    "MIDPOINT": MIDPOINT,
    "MIDPRICE": MIDPRICE,
    "PivotPoints": PivotPoints,
    "SAR": SAR,
    "SAREXT": SAREXT,
    "SMA": SMA,
    "T3": T3,
    "TEMA": TEMA,
    "TRIMA": TRIMA,
    "WMA": WMA,

    # Volatility indicators
    "ATR": ATR,

    # Volume indicators
    "AD": AD,
    "ADOSC": ADOSC,
    "OBV": OBV,
}


def get_indicator_class(indicator_type: str) -> Type[BaseIndicator]:
    """
    Get indicator class by name.

    Args:
        indicator_type: Indicator type name (e.g., "RSI", "MACD", "SMA")

    Returns:
        Indicator class

    Raises:
        ValueError: If indicator type is unknown

    Example:
        >>> indicator_class = get_indicator_class("RSI")
        >>> # indicator_class is RSI class
    """
    if indicator_type not in INDICATOR_REGISTRY:
        available = ", ".join(sorted(INDICATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown indicator type: '{indicator_type}'. "
            f"Available indicators: {available}"
        )

    return INDICATOR_REGISTRY[indicator_type]


def create_indicator(
    frame,
    indicator_type: str,
    **kwargs
) -> BaseIndicator:
    """
    Create an indicator instance from textual name.

    Args:
        frame: Frame instance to bind the indicator to
        indicator_type: Indicator type name (e.g., "RSI", "MACD", "SMA")
        **kwargs: Indicator-specific parameters (e.g., length=14 for RSI)

    Returns:
        Instantiated indicator

    Raises:
        ValueError: If indicator type is unknown
        TypeError: If invalid parameters provided

    Example:
        >>> rsi = create_indicator(frame, "RSI", length=14)
        >>> macd = create_indicator(frame, "MACD", fast=12, slow=26, signal=9)
    """
    indicator_class = get_indicator_class(indicator_type)
    return indicator_class(frame=frame, **kwargs)


def list_available_indicators() -> List[str]:
    """
    Get list of all available indicator names.

    Returns:
        Sorted list of indicator type names

    Example:
        >>> indicators = list_available_indicators()
        >>> print(indicators)
        ['ATR', 'BB', 'BollingerBands', 'EMA', 'FVG', 'MACD', ...]
    """
    return sorted(INDICATOR_REGISTRY.keys())


def register_indicator(name: str, indicator_class: Type[BaseIndicator]) -> None:
    """
    Register a custom indicator in the registry.

    Useful for adding user-defined indicators to the textual API.

    Args:
        name: Name to register the indicator under
        indicator_class: Indicator class (must inherit from BaseIndicator)

    Raises:
        ValueError: If name already exists
        TypeError: If indicator_class doesn't inherit from BaseIndicator

    Example:
        >>> class MyCustomIndicator(BaseIndicator):
        ...     pass
        >>> register_indicator("CUSTOM", MyCustomIndicator)
    """
    if name in INDICATOR_REGISTRY:
        raise ValueError(f"Indicator '{name}' is already registered")

    if not issubclass(indicator_class, BaseIndicator):
        raise TypeError(f"Indicator class must inherit from BaseIndicator")

    INDICATOR_REGISTRY[name] = indicator_class
