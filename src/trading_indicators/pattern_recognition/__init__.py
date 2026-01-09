"""
Candlestick Pattern Recognition Indicators

This module provides 61 candlestick pattern recognition functions from TA-Lib.
All patterns return integer values:
- 0: Pattern not detected
- 100: Bullish pattern detected
- -100: Bearish pattern detected

Each pattern can be used in two modes:
1. Auto-sync mode: Automatically updates when frame receives new candles
2. Utility mode: Static compute() method for quick pattern detection
"""

from .cdl2crows import CDL2CROWS
from .cdl3blackcrows import CDL3BLACKCROWS
from .cdl3inside import CDL3INSIDE
from .cdl3linestrike import CDL3LINESTRIKE
from .cdl3outside import CDL3OUTSIDE
from .cdl3starsinsouth import CDL3STARSINSOUTH
from .cdl3whitesoldiers import CDL3WHITESOLDIERS
from .cdlabandonedbaby import CDLABANDONEDBABY
from .cdladvanceblock import CDLADVANCEBLOCK
from .cdlbelthold import CDLBELTHOLD
from .cdlbreakaway import CDLBREAKAWAY
from .cdlclosingmarubozu import CDLCLOSINGMARUBOZU
from .cdlconcealbabyswall import CDLCONCEALBABYSWALL
from .cdlcounterattack import CDLCOUNTERATTACK
from .cdldarkcloudcover import CDLDARKCLOUDCOVER
from .cdldoji import CDLDOJI
from .cdldojistar import CDLDOJISTAR
from .cdldragonflydoji import CDLDRAGONFLYDOJI
from .cdlengulfing import CDLENGULFING
from .cdleveningdojistar import CDLEVENINGDOJISTAR
from .cdleveningstar import CDLEVENINGSTAR
from .cdlgapsidesidewhite import CDLGAPSIDESIDEWHITE
from .cdlgravestonedoji import CDLGRAVESTONEDOJI
from .cdlhammer import CDLHAMMER
from .cdlhangingman import CDLHANGINGMAN
from .cdlharami import CDLHARAMI
from .cdlharamicross import CDLHARAMICROSS
from .cdlhighwave import CDLHIGHWAVE
from .cdlhikkake import CDLHIKKAKE
from .cdlhikkakemod import CDLHIKKAKEMOD
from .cdlhomingpigeon import CDLHOMINGPIGEON
from .cdlidentical3crows import CDLIDENTICAL3CROWS
from .cdlinneck import CDLINNECK
from .cdlinvertedhammer import CDLINVERTEDHAMMER
from .cdlkicking import CDLKICKING
from .cdlkickingbylength import CDLKICKINGBYLENGTH
from .cdlladderbottom import CDLLADDERBOTTOM
from .cdllongleggeddoji import CDLLONGLEGGEDDOJI
from .cdllongline import CDLLONGLINE
from .cdlmarubozu import CDLMARUBOZU
from .cdlmatchinglow import CDLMATCHINGLOW
from .cdlmathold import CDLMATHOLD
from .cdlmorningdojistar import CDLMORNINGDOJISTAR
from .cdlmorningstar import CDLMORNINGSTAR
from .cdlonneck import CDLONNECK
from .cdlpiercing import CDLPIERCING
from .cdlrickshawman import CDLRICKSHAWMAN
from .cdlrisefall3methods import CDLRISEFALL3METHODS
from .cdlseparatinglines import CDLSEPARATINGLINES
from .cdlshootingstar import CDLSHOOTINGSTAR
from .cdlshortline import CDLSHORTLINE
from .cdlspinningtop import CDLSPINNINGTOP
from .cdlstalledpattern import CDLSTALLEDPATTERN
from .cdlsticksandwich import CDLSTICKSANDWICH
from .cdltakuri import CDLTAKURI
from .cdltasukigap import CDLTASUKIGAP
from .cdlthrusting import CDLTHRUSTING
from .cdltristar import CDLTRISTAR
from .cdlunique3river import CDLUNIQUE3RIVER
from .cdlupsidegap2crows import CDLUPSIDEGAP2CROWS
from .cdlxsidegap3methods import CDLXSIDEGAP3METHODS

__all__ = [
    'CDL2CROWS',
    'CDL3BLACKCROWS',
    'CDL3INSIDE',
    'CDL3LINESTRIKE',
    'CDL3OUTSIDE',
    'CDL3STARSINSOUTH',
    'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY',
    'CDLADVANCEBLOCK',
    'CDLBELTHOLD',
    'CDLBREAKAWAY',
    'CDLCLOSINGMARUBOZU',
    'CDLCONCEALBABYSWALL',
    'CDLCOUNTERATTACK',
    'CDLDARKCLOUDCOVER',
    'CDLDOJI',
    'CDLDOJISTAR',
    'CDLDRAGONFLYDOJI',
    'CDLENGULFING',
    'CDLEVENINGDOJISTAR',
    'CDLEVENINGSTAR',
    'CDLGAPSIDESIDEWHITE',
    'CDLGRAVESTONEDOJI',
    'CDLHAMMER',
    'CDLHANGINGMAN',
    'CDLHARAMI',
    'CDLHARAMICROSS',
    'CDLHIGHWAVE',
    'CDLHIKKAKE',
    'CDLHIKKAKEMOD',
    'CDLHOMINGPIGEON',
    'CDLIDENTICAL3CROWS',
    'CDLINNECK',
    'CDLINVERTEDHAMMER',
    'CDLKICKING',
    'CDLKICKINGBYLENGTH',
    'CDLLADDERBOTTOM',
    'CDLLONGLEGGEDDOJI',
    'CDLLONGLINE',
    'CDLMARUBOZU',
    'CDLMATCHINGLOW',
    'CDLMATHOLD',
    'CDLMORNINGDOJISTAR',
    'CDLMORNINGSTAR',
    'CDLONNECK',
    'CDLPIERCING',
    'CDLRICKSHAWMAN',
    'CDLRISEFALL3METHODS',
    'CDLSEPARATINGLINES',
    'CDLSHOOTINGSTAR',
    'CDLSHORTLINE',
    'CDLSPINNINGTOP',
    'CDLSTALLEDPATTERN',
    'CDLSTICKSANDWICH',
    'CDLTAKURI',
    'CDLTASUKIGAP',
    'CDLTHRUSTING',
    'CDLTRISTAR',
    'CDLUNIQUE3RIVER',
    'CDLUPSIDEGAP2CROWS',
    'CDLXSIDEGAP3METHODS',
]
