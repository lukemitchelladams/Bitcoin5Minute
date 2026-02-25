"""
PolyCopy BTC Bot — 5-Minute Bitcoin Prediction Dashboard
=========================================================
Data sources:
  • Binance REST API   — live OHLCV candles, order book, 24h stats
  • Alternative.me     — Crypto Fear & Greed Index
  • CryptoPanic        — live crypto news sentiment
  • Polymarket API     — active BTC prediction markets

Indicators computed:
  RSI-14, MACD(12/26/9), Bollinger Bands(20/2),
  EMA-9/21/50, VWAP, ATR-14, OBV, Stochastic-14,
  Order book imbalance, Volume delta

Run:
  streamlit run btc_bot.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import time
import json

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Prediction Bot",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #05080f;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(255,140,0,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(0,201,255,0.04) 0%, transparent 60%);
}
section[data-testid="stSidebar"] {
    background: #080c14 !important;
    border-right: 1px solid #0f1825;
}
section[data-testid="stSidebar"] * { color: #8a9bb0 !important; }
section[data-testid="stSidebar"] h3 { color: #f0f8ff !important; }

[data-testid="metric-container"] {
    background: #080c14;
    border: 1px solid #0f1825;
    border-radius: 10px;
    padding: 14px !important;
}
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 8px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #2a3a4c !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 20px !important;
    color: #f0f8ff !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff8c00, #ff6b00) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: .8px !important;
    padding: 10px 24px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ffaa00, #ff8c00) !important;
    box-shadow: 0 4px 20px rgba(255,140,0,0.35) !important;
}

/* Signal card */
.sig-card {
    background: #080c14;
    border: 1px solid #0f1825;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.sig-title {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #2a3a4c;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.sig-value {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    color: #f0f8ff;
}
.sig-value.bull { color: #00e676; }
.sig-value.bear { color: #ff5252; }
.sig-value.neut { color: #ff8c00; }

/* Big prediction box */
.pred-box {
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-bottom: 20px;
    border: 2px solid;
}
.pred-box.bull {
    background: rgba(0,230,118,0.06);
    border-color: rgba(0,230,118,0.3);
}
.pred-box.bear {
    background: rgba(255,82,82,0.06);
    border-color: rgba(255,82,82,0.3);
}
.pred-box.neut {
    background: rgba(255,140,0,0.06);
    border-color: rgba(255,140,0,0.3);
}
.pred-direction {
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.pred-direction.bull { color: #00e676; }
.pred-direction.bear { color: #ff5252; }
.pred-direction.neut { color: #ff8c00; }
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5a7a8c;
    margin-bottom: 14px;
}
.pred-conf {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
}
.pred-conf.bull { color: #00e676; }
.pred-conf.bear { color: #ff5252; }
.pred-conf.neut { color: #ff8c00; }

/* Bar chart for signal breakdown */
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}
.bar-name {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #5a7a8c;
    width: 80px;
    flex-shrink: 0;
    text-transform: uppercase;
}
.bar-track {
    flex: 1;
    height: 6px;
    background: #0f1825;
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill-bull { height: 100%; border-radius: 3px; background: #00e676; }
.bar-fill-bear { height: 100%; border-radius: 3px; background: #ff5252; }
.bar-fill-neut { height: 100%; border-radius: 3px; background: #ff8c00; }
.bar-score {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #5a7a8c;
    width: 30px;
    text-align: right;
    flex-shrink: 0;
}

/* News item */
.news-item {
    background: #080c14;
    border: 1px solid #0f1825;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 7px;
}
.news-title { font-size: 12px; font-weight: 600; color: #c8d8e8; line-height: 1.4; margin-bottom: 4px; }
.news-meta { font-family: 'Space Mono', monospace; font-size: 9px; color: #2a3a4c; }

/* Polymarket card */
.pm-card {
    background: #080c14;
    border: 1px solid #0f1825;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.pm-question { font-size: 13px; font-weight: 600; color: #f0f8ff; margin-bottom: 10px; line-height: 1.4; }
.pm-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }
.pm-metric-label { font-family: 'Space Mono', monospace; font-size: 8px; color: #2a3a4c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.pm-metric-value { font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700; color: #f0f8ff; }
.pm-align-bull { color: #00e676 !important; }
.pm-align-bear { color: #ff5252 !important; }
.pm-align-neut { color: #ff8c00 !important; }

.header-logo {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #f0f8ff;
}
.header-logo span { color: #ff8c00; }
.header-sub {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #2a3a4c;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 2px;
    margin-bottom: 20px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #0f1825; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  API FUNCTIONS
# ─────────────────────────────────────────────────────────────────
KRAKEN   = "https://api.kraken.com/0/public"
HEADERS  = {"Accept": "application/json"}

@st.cache_data(ttl=30)
def fetch_candles(symbol="XBTUSD", interval=5, limit=200) -> pd.DataFrame:
    """Fetch OHLCV candles from Kraken (no geo-restrictions)."""
    try:
        r = requests.get(f"{KRAKEN}/OHLC",
                         params={"pair": symbol, "interval": interval},
                         headers=HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            st.error(f"Kraken error: {data['error']}")
            return pd.DataFrame()
        result = data.get("result", {})
        key    = list(result.keys())[0]
        raw    = result[key][-limit:]
        df = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","vwap","volume","trades"
        ])
        for col in ["open","high","low","close","vwap","volume"]:
            df[col] = pd.to_numeric(df[col])
        df["ts"]             = pd.to_datetime(df["ts"], unit="s", utc=True)
        df["taker_buy_base"] = df["volume"] * 0.52   # estimate ~52% buy side
        df["quote_vol"]      = df["volume"] * df["close"]
        df["taker_buy_quote"]= df["taker_buy_base"] * df["close"]
        return df
    except Exception as e:
        st.error(f"Kraken candles error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_ticker(symbol="XBTUSD") -> dict:
    """Fetch ticker stats from Kraken."""
    try:
        r = requests.get(f"{KRAKEN}/Ticker",
                         params={"pair": symbol}, headers=HEADERS, timeout=8)
        r.raise_for_status()
        data   = r.json()
        result = data.get("result", {})
        key    = list(result.keys())[0]
        t      = result[key]
        return {
            "lastPrice":           t["c"][0],
            "priceChangePercent":  str(round((float(t["c"][0]) - float(t["o"])) / float(t["o"]) * 100, 2)),
            "highPrice":           t["h"][1],
            "lowPrice":            t["l"][1],
            "quoteVolume":         str(float(t["v"][1]) * float(t["c"][0])),
        }
    except Exception:
        return {}

@st.cache_data(ttl=10)
def fetch_orderbook(symbol="XBTUSD", limit=50) -> dict:
    """Fetch order book depth from Kraken."""
    try:
        r = requests.get(f"{KRAKEN}/Depth",
                         params={"pair": symbol, "count": limit},
                         headers=HEADERS, timeout=8)
        r.raise_for_status()
        data   = r.json()
        result = data.get("result", {})
        key    = list(result.keys())[0]
        book   = result[key]
        bids   = [(float(p), float(q)) for p, q, _ in book.get("bids", [])]
        asks   = [(float(p), float(q)) for p, q, _ in book.get("asks", [])]
        return {"bids": bids, "asks": asks}
    except Exception:
        return {"bids": [], "asks": []}

@st.cache_data(ttl=300)
def fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=2", timeout=8)
        r.raise_for_status()
        data = r.json().get("data", [{}])
        return {
            "value":       int(data[0].get("value", 50)),
            "label":       data[0].get("value_classification", "Neutral"),
            "prev_value":  int(data[1].get("value", 50)) if len(data) > 1 else 50,
        }
    except Exception:
        return {"value": 50, "label": "Neutral", "prev_value": 50}

@st.cache_data(ttl=120)
def fetch_news() -> list:
    """Fetch crypto news from CryptoPanic (public, no key needed for basic)."""
    try:
        r = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": "public", "currencies": "BTC", "kind": "news", "public": "true"},
            timeout=10
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:8]
        items = []
        for item in results:
            votes  = item.get("votes", {})
            bull   = votes.get("positive", 0)
            bear   = votes.get("negative", 0)
            total  = bull + bear
            sent   = "bull" if bull > bear else ("bear" if bear > bull else "neut")
            items.append({
                "title": item.get("title", ""),
                "url":   item.get("url", "#"),
                "published": item.get("published_at", ""),
                "sentiment": sent,
                "bull": bull, "bear": bear, "total": total,
            })
        return items
    except Exception:
        return []

@st.cache_data(ttl=60)
def fetch_polymarket_btc() -> list:
    """Fetch active BTC-related markets from Polymarket."""
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 100},
            headers=HEADERS, timeout=12
        )
        r.raise_for_status()
        data = r.json()
        markets = data if isinstance(data, list) else data.get("markets", [])
        btc = []
        for m in markets:
            q = (m.get("question") or m.get("title") or "").lower()
            if "bitcoin" in q or "btc" in q:
                btc.append(m)
        return btc[:10]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50:
        return {}

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    # ── RSI ────────────────────────────────────────────
    def rsi(src, period=14):
        delta = np.diff(src)
        gains = np.where(delta > 0, delta, 0.0)
        losses= np.where(delta < 0, -delta, 0.0)
        avg_g = np.convolve(gains,  np.ones(period)/period, mode='valid')
        avg_l = np.convolve(losses, np.ones(period)/period, mode='valid')
        rs    = np.where(avg_l == 0, 100, avg_g / avg_l)
        return 100 - (100 / (1 + rs))

    rsi_vals = rsi(close)
    rsi_now  = rsi_vals[-1] if len(rsi_vals) else 50

    # ── EMA ────────────────────────────────────────────
    def ema(src, span):
        s = pd.Series(src)
        return s.ewm(span=span, adjust=False).mean().values

    ema9  = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)

    # ── MACD ───────────────────────────────────────────
    fast   = ema(close, 12)
    slow   = ema(close, 26)
    macd_l = fast - slow
    signal = ema(macd_l, 9)
    hist   = macd_l - signal

    # ── Bollinger Bands ────────────────────────────────
    period = 20
    s      = pd.Series(close)
    sma20  = s.rolling(period).mean().values
    std20  = s.rolling(period).std().values
    bb_up  = sma20 + 2 * std20
    bb_dn  = sma20 - 2 * std20
    bb_pos = (close[-1] - bb_dn[-1]) / (bb_up[-1] - bb_dn[-1] + 1e-9)

    # ── Stochastic ─────────────────────────────────────
    def stoch(h, l, c, k=14, d=3):
        lo  = pd.Series(l).rolling(k).min().values
        hi  = pd.Series(h).rolling(k).max().values
        k_  = 100 * (c - lo) / (hi - lo + 1e-9)
        d_  = pd.Series(k_).rolling(d).mean().values
        return k_, d_

    stoch_k, stoch_d = stoch(high, low, close)

    # ── ATR ────────────────────────────────────────────
    tr   = np.maximum(high[1:]-low[1:],
           np.maximum(np.abs(high[1:]-close[:-1]),
                      np.abs(low[1:]-close[:-1])))
    atr  = pd.Series(tr).rolling(14).mean().values

    # ── VWAP ───────────────────────────────────────────
    typical = (high + low + close) / 3
    vwap    = (typical * volume).cumsum() / (volume.cumsum() + 1e-9)

    # ── OBV ────────────────────────────────────────────
    delta = np.diff(close)
    sign  = np.sign(delta)
    obv   = np.cumsum(np.concatenate([[0], sign * volume[1:]]))

    # ── Volume delta ───────────────────────────────────
    buy_vol  = df["taker_buy_base"].values
    sell_vol = volume - buy_vol
    vol_delta = buy_vol[-1] - sell_vol[-1]

    return {
        "close":     close[-1],
        "prev_close":close[-2],
        # RSI
        "rsi":       float(rsi_now),
        # EMA
        "ema9":      float(ema9[-1]),
        "ema21":     float(ema21[-1]),
        "ema50":     float(ema50[-1]),
        # MACD
        "macd":      float(macd_l[-1]),
        "macd_sig":  float(signal[-1]),
        "macd_hist": float(hist[-1]),
        # BB
        "bb_upper":  float(bb_up[-1]),
        "bb_lower":  float(bb_dn[-1]),
        "bb_mid":    float(sma20[-1]),
        "bb_pos":    float(bb_pos),    # 0=at lower, 1=at upper
        # Stoch
        "stoch_k":   float(stoch_k[-1]),
        "stoch_d":   float(stoch_d[-1]),
        # ATR
        "atr":       float(atr[-1]) if not np.isnan(atr[-1]) else 0,
        # VWAP
        "vwap":      float(vwap[-1]),
        # OBV trend (last 5 bars)
        "obv_slope": float(np.polyfit(range(5), obv[-5:], 1)[0]),
        # Volume
        "vol_delta":   float(vol_delta),
        "vol_ratio":   float(buy_vol[-1] / (sell_vol[-1] + 1e-9)),
        # Raw series for chart
        "df":        df,
        "ema9_s":    ema9,
        "ema21_s":   ema21,
        "ema50_s":   ema50,
        "bb_up_s":   bb_up,
        "bb_dn_s":   bb_dn,
        "macd_s":    macd_l,
        "signal_s":  signal,
        "hist_s":    hist,
        "rsi_s":     np.concatenate([np.full(len(close)-len(rsi_vals), np.nan), rsi_vals]),
    }

# ─────────────────────────────────────────────────────────────────
#  ORDER BOOK ANALYSIS
# ─────────────────────────────────────────────────────────────────
def analyze_orderbook(ob: dict) -> dict:
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return {"imbalance": 0, "bid_wall": 0, "ask_wall": 0, "spread": 0, "signal": "neut"}

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    total   = bid_vol + ask_vol + 1e-9
    imbal   = (bid_vol - ask_vol) / total   # +1 = all bids, -1 = all asks

    # Largest single walls (in BTC)
    bid_wall = max(q for _, q in bids)
    ask_wall = max(q for _, q in asks)

    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread   = best_ask - best_bid

    signal = "bull" if imbal > 0.1 else ("bear" if imbal < -0.1 else "neut")
    return {
        "imbalance": round(imbal * 100, 1),
        "bid_vol":   round(bid_vol, 2),
        "ask_vol":   round(ask_vol, 2),
        "bid_wall":  round(bid_wall, 3),
        "ask_wall":  round(ask_wall, 3),
        "spread":    round(spread, 2),
        "signal":    signal,
    }

# ─────────────────────────────────────────────────────────────────
#  SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────
def analyze_sentiment(news: list, fg: dict) -> dict:
    bull_news = sum(1 for n in news if n["sentiment"] == "bull")
    bear_news = sum(1 for n in news if n["sentiment"] == "bear")
    total     = len(news) or 1
    news_score = (bull_news - bear_news) / total   # -1 to +1

    fg_val = fg.get("value", 50)
    # FG 0-25=Extreme Fear, 25-45=Fear, 45-55=Neutral, 55-75=Greed, 75-100=Extreme Greed
    fg_score = (fg_val - 50) / 50   # -1 to +1

    combined = (news_score * 0.4 + fg_score * 0.6)
    signal   = "bull" if combined > 0.1 else ("bear" if combined < -0.1 else "neut")

    return {
        "fg_value":   fg_val,
        "fg_label":   fg.get("label", "Neutral"),
        "fg_prev":    fg.get("prev_value", 50),
        "news_bull":  bull_news,
        "news_bear":  bear_news,
        "news_score": round(news_score * 100, 1),
        "combined":   round(combined * 100, 1),
        "signal":     signal,
    }

# ─────────────────────────────────────────────────────────────────
#  PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
def generate_prediction(ind: dict, ob_analysis: dict, sent: dict) -> dict:
    if not ind:
        return {"direction": "NEUTRAL", "confidence": 0, "signals": {}, "cls": "neut"}

    signals = {}

    # ── RSI ──────────────────────────────────────────
    rsi = ind["rsi"]
    if rsi < 30:
        signals["RSI"] = ("BULL", 85, "Oversold")
    elif rsi > 70:
        signals["RSI"] = ("BEAR", 85, "Overbought")
    elif rsi < 45:
        signals["RSI"] = ("BULL", 55, "Below midline")
    elif rsi > 55:
        signals["RSI"] = ("BEAR", 55, "Above midline")
    else:
        signals["RSI"] = ("NEUT", 0, "Neutral zone")

    # ── MACD ─────────────────────────────────────────
    hist = ind["macd_hist"]
    macd = ind["macd"]
    if hist > 0 and macd > ind["macd_sig"]:
        signals["MACD"] = ("BULL", 70, "Bullish crossover")
    elif hist < 0 and macd < ind["macd_sig"]:
        signals["MACD"] = ("BEAR", 70, "Bearish crossover")
    elif hist > 0:
        signals["MACD"] = ("BULL", 45, "Positive histogram")
    else:
        signals["MACD"] = ("BEAR", 45, "Negative histogram")

    # ── Bollinger Bands ───────────────────────────────
    bb_pos = ind["bb_pos"]
    close  = ind["close"]
    if bb_pos < 0.1:
        signals["BB"] = ("BULL", 75, "Near lower band")
    elif bb_pos > 0.9:
        signals["BB"] = ("BEAR", 75, "Near upper band")
    elif close > ind["bb_mid"]:
        signals["BB"] = ("BULL", 40, "Above midline")
    else:
        signals["BB"] = ("BEAR", 40, "Below midline")

    # ── EMA Stack ────────────────────────────────────
    e9, e21, e50 = ind["ema9"], ind["ema21"], ind["ema50"]
    if e9 > e21 > e50:
        signals["EMA"] = ("BULL", 80, "Bullish stack 9>21>50")
    elif e9 < e21 < e50:
        signals["EMA"] = ("BEAR", 80, "Bearish stack 9<21<50")
    elif e9 > e21:
        signals["EMA"] = ("BULL", 50, "Short-term bullish")
    else:
        signals["EMA"] = ("BEAR", 50, "Short-term bearish")

    # ── Stochastic ───────────────────────────────────
    sk, sd = ind["stoch_k"], ind["stoch_d"]
    if sk < 20 and sd < 20:
        signals["Stoch"] = ("BULL", 78, "Oversold territory")
    elif sk > 80 and sd > 80:
        signals["Stoch"] = ("BEAR", 78, "Overbought territory")
    elif sk > sd:
        signals["Stoch"] = ("BULL", 45, "K above D")
    else:
        signals["Stoch"] = ("BEAR", 45, "K below D")

    # ── VWAP ─────────────────────────────────────────
    vwap = ind["vwap"]
    if close > vwap * 1.002:
        signals["VWAP"] = ("BULL", 65, f"Above VWAP ${vwap:,.0f}")
    elif close < vwap * 0.998:
        signals["VWAP"] = ("BEAR", 65, f"Below VWAP ${vwap:,.0f}")
    else:
        signals["VWAP"] = ("NEUT", 0, f"At VWAP ${vwap:,.0f}")

    # ── OBV ──────────────────────────────────────────
    obv_slope = ind["obv_slope"]
    if obv_slope > 0:
        signals["OBV"] = ("BULL", 60, "Rising volume trend")
    else:
        signals["OBV"] = ("BEAR", 60, "Falling volume trend")

    # ── Volume Delta ─────────────────────────────────
    vd = ind["vol_delta"]
    vr = ind["vol_ratio"]
    if vr > 1.3:
        signals["Vol Delta"] = ("BULL", 70, f"Buy pressure {vr:.1f}x")
    elif vr < 0.7:
        signals["Vol Delta"] = ("BEAR", 70, f"Sell pressure {1/vr:.1f}x")
    else:
        signals["Vol Delta"] = ("NEUT", 0, "Balanced volume")

    # ── Order Book ───────────────────────────────────
    imbal = ob_analysis.get("imbalance", 0)
    if imbal > 10:
        signals["Order Book"] = ("BULL", 72, f"Bid-heavy +{imbal:.1f}%")
    elif imbal < -10:
        signals["Order Book"] = ("BEAR", 72, f"Ask-heavy {imbal:.1f}%")
    else:
        signals["Order Book"] = ("NEUT", 0, f"Balanced {imbal:.1f}%")

    # ── Sentiment ────────────────────────────────────
    fg = sent.get("fg_value", 50)
    if fg < 25:
        signals["Sentiment"] = ("BULL", 65, f"Extreme Fear → contrarian buy")
    elif fg > 75:
        signals["Sentiment"] = ("BEAR", 65, f"Extreme Greed → contrarian sell")
    elif sent["signal"] == "bull":
        signals["Sentiment"] = ("BULL", 45, f"Bullish news flow")
    elif sent["signal"] == "bear":
        signals["Sentiment"] = ("BEAR", 45, f"Bearish news flow")
    else:
        signals["Sentiment"] = ("NEUT", 0, "Neutral sentiment")

    # ── AGGREGATE ────────────────────────────────────
    bull_score = sum(s[1] for s in signals.values() if s[0] == "BULL")
    bear_score = sum(s[1] for s in signals.values() if s[0] == "BEAR")
    total      = bull_score + bear_score + 1e-9

    if bull_score > bear_score:
        direction  = "LONG  ▲"
        confidence = round(bull_score / total * 100)
        cls        = "bull"
    elif bear_score > bull_score:
        direction  = "SHORT  ▼"
        confidence = round(bear_score / total * 100)
        cls        = "bear"
    else:
        direction  = "NEUTRAL"
        confidence = 50
        cls        = "neut"

    return {
        "direction":   direction,
        "confidence":  confidence,
        "bull_score":  round(bull_score),
        "bear_score":  round(bear_score),
        "signals":     signals,
        "cls":         cls,
    }

# ─────────────────────────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────────────────────────
def build_chart(ind: dict) -> go.Figure:
    df = ind.get("df")
    if df is None or df.empty:
        return go.Figure()

    n = len(df)
    ts = df["ts"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=["", "MACD", "RSI"]
    )

    # ── Candlesticks ──────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=ts, open=df["open"], high=df["high"],
        low=df["low"],  close=df["close"],
        name="BTC/USDT",
        increasing_fillcolor="#00e676",
        increasing_line_color="#00e676",
        decreasing_fillcolor="#ff5252",
        decreasing_line_color="#ff5252",
    ), row=1, col=1)

    # EMAs
    for vals, col, nm in [(ind["ema9_s"],"#ff8c00","EMA9"),
                           (ind["ema21_s"],"#00c9ff","EMA21"),
                           (ind["ema50_s"],"#b967ff","EMA50")]:
        fig.add_trace(go.Scatter(x=ts, y=vals, name=nm,
                                  line=dict(color=col, width=1.2)), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=ts, y=ind["bb_up_s"], name="BB Upper",
                              line=dict(color="rgba(100,150,255,0.4)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=ind["bb_dn_s"], name="BB Lower",
                              fill="tonexty",
                              fillcolor="rgba(100,150,255,0.04)",
                              line=dict(color="rgba(100,150,255,0.4)", width=1, dash="dot")), row=1, col=1)

    # Volume bars
    colors = ["#00e676" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ff5252" for i in range(n)]
    fig.add_trace(go.Bar(x=ts, y=df["volume"], name="Volume",
                          marker_color=[c.replace(")", ",0.4)").replace("rgb","rgba") if c.startswith("rgb") else c + "66" for c in colors],
                          yaxis="y3"), row=1, col=1)

    # ── MACD ──────────────────────────────────────────
    hist_colors = ["#00e676" if v >= 0 else "#ff5252" for v in ind["hist_s"]]
    fig.add_trace(go.Bar(x=ts, y=ind["hist_s"], name="MACD Hist",
                          marker_color=hist_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=ind["macd_s"], name="MACD",
                              line=dict(color="#ff8c00", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=ind["signal_s"], name="Signal",
                              line=dict(color="#00c9ff", width=1.2)), row=2, col=1)

    # ── RSI ───────────────────────────────────────────
    fig.add_trace(go.Scatter(x=ts, y=ind["rsi_s"], name="RSI",
                              line=dict(color="#b967ff", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,82,82,0.4)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,230,118,0.4)", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.1)", row=3, col=1)

    fig.update_layout(
        height=560,
        plot_bgcolor="#080c14",
        paper_bgcolor="#05080f",
        font=dict(family="Space Mono, monospace", size=10, color="#5a7a8c"),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=9,
                     bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#0f1825", zeroline=False, row=i, col=1,
            showticklabels=(i == 3)
        )
        fig.update_yaxes(gridcolor="#0f1825", zeroline=False, row=i, col=1)

    return fig

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Settings")
    st.markdown("---")

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    candle_limit = st.slider("Candles to load", 100, 500, 200, 50)

    st.markdown("---")
    st.markdown("### 📊 Active Indicators")
    use_rsi   = st.checkbox("RSI-14",           value=True)
    use_macd  = st.checkbox("MACD (12/26/9)",   value=True)
    use_bb    = st.checkbox("Bollinger Bands",  value=True)
    use_ema   = st.checkbox("EMA 9/21/50",      value=True)
    use_stoch = st.checkbox("Stochastic-14",    value=True)
    use_vwap  = st.checkbox("VWAP",             value=True)
    use_obv   = st.checkbox("OBV",              value=True)
    use_vol   = st.checkbox("Volume Delta",     value=True)
    use_ob    = st.checkbox("Order Book Depth", value=True)
    use_sent  = st.checkbox("Sentiment / News", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2a3a4c;line-height:1.9;">
    ⚠ FOR EDUCATIONAL USE ONLY<br>
    This is not financial advice.<br>
    Never bet more than you<br>
    can afford to lose.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="header-logo">₿TC <span>Prediction Bot</span></div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">5-MINUTE SIGNAL SCANNER · POLYMARKET INTEGRATION · LIVE DATA</div>', unsafe_allow_html=True)

col_btn, col_time, _ = st.columns([1, 2, 5])
with col_btn:
    refresh = st.button("⟳ Refresh Now", use_container_width=True)
with col_time:
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#2a3a4c;padding-top:12px;">
    {datetime.now(timezone.utc).strftime("LAST UPDATE  %H:%M:%S UTC")}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
#  FETCH ALL DATA
# ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching live data..."):
    df_5m    = fetch_candles("XBTUSD", 5, candle_limit)
    ticker   = fetch_ticker("XBTUSD")
    ob       = fetch_orderbook("XBTUSD", 100)
    fg       = fetch_fear_greed()
    news     = fetch_news() if use_sent else []
    pm_mkts  = fetch_polymarket_btc()

ind        = compute_indicators(df_5m)
ob_data    = analyze_orderbook(ob)
sent_data  = analyze_sentiment(news, fg)
prediction = generate_prediction(
    ind if use_rsi else {},
    ob_data  if use_ob   else {},
    sent_data if use_sent else {}
)

# Current price info
price     = float(ticker.get("lastPrice", ind.get("close", 0)))
price_chg = float(ticker.get("priceChangePercent", 0))
vol_24h   = float(ticker.get("quoteVolume", 0))
high_24h  = float(ticker.get("highPrice", 0))
low_24h   = float(ticker.get("lowPrice", 0))

# ─────────────────────────────────────────────────────────────────
#  TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("BTC/USDT",     f"${price:,.2f}",  f"{price_chg:+.2f}%")
with m2: st.metric("24h High",     f"${high_24h:,.0f}")
with m3: st.metric("24h Low",      f"${low_24h:,.0f}")
with m4: st.metric("24h Volume",   f"${vol_24h/1e9:.2f}B")
with m5: st.metric("Fear & Greed", f"{fg['value']} — {fg['label']}", f"{fg['value']-fg['prev_value']:+d}")
with m6: st.metric("OB Imbalance", f"{ob_data['imbalance']:+.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
#  MAIN LAYOUT  — Chart left, Prediction right
# ─────────────────────────────────────────────────────────────────
chart_col, pred_col = st.columns([3, 1])

with chart_col:
    st.markdown("#### 5m BTC/USDT · Candlestick Chart")
    if ind:
        st.plotly_chart(build_chart(ind), use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("Could not load chart data.")

with pred_col:
    st.markdown("#### ⚡ 5-Min Prediction")

    # Big prediction box
    cls = prediction["cls"]
    dir_sym = "▲" if cls == "bull" else ("▼" if cls == "bear" else "◆")
    dir_lbl = "LONG" if cls == "bull" else ("SHORT" if cls == "bear" else "NEUTRAL")

    st.markdown(f"""
    <div class="pred-box {cls}">
      <div class="pred-direction {cls}">{dir_sym}</div>
      <div class="pred-label">{dir_lbl} SIGNAL</div>
      <div class="pred-conf {cls}">{prediction['confidence']}%</div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2a3a4c;margin-top:6px;">CONFIDENCE SCORE</div>
    </div>
    """, unsafe_allow_html=True)

    # Score bars
    bull_sc = prediction.get('bull_score', 0)
    bear_sc = prediction.get('bear_score', 0)
    st.markdown(f"""
    <div style="margin-bottom:12px;">
      <div class="bar-row">
        <span class="bar-name">BULL</span>
        <div class="bar-track"><div class="bar-fill-bull" style="width:{min(bull_sc/8,100)}%"></div></div>
        <span class="bar-score">{bull_sc}</span>
      </div>
      <div class="bar-row">
        <span class="bar-name">BEAR</span>
        <div class="bar-track"><div class="bar-fill-bear" style="width:{min(bear_sc/8,100)}%"></div></div>
        <span class="bar-score">{bear_sc}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Individual signals
    st.markdown("**Signal Breakdown**")
    for name, (sig, strength, desc) in prediction["signals"].items():
        cls2 = "bull" if sig == "BULL" else ("bear" if sig == "BEAR" else "neut")
        icon = "▲" if sig == "BULL" else ("▼" if sig == "BEAR" else "◆")
        st.markdown(f"""
        <div class="sig-card">
          <div class="sig-title">{name}</div>
          <div class="sig-value {cls2}">{icon} {desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  INDICATOR VALUES TABLE
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📐 Indicator Values")

if ind:
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    with i1: st.metric("RSI-14",      f"{ind['rsi']:.1f}",    delta=None)
    with i2: st.metric("MACD Hist",   f"{ind['macd_hist']:.1f}")
    with i3: st.metric("BB Position", f"{ind['bb_pos']*100:.0f}%")
    with i4: st.metric("Stoch %K",    f"{ind['stoch_k']:.1f}")
    with i5: st.metric("VWAP",        f"${ind['vwap']:,.0f}")
    with i6: st.metric("ATR-14",      f"${ind['atr']:,.0f}")

    i7, i8, i9, i10, i11, i12 = st.columns(6)
    with i7:  st.metric("EMA-9",  f"${ind['ema9']:,.0f}")
    with i8:  st.metric("EMA-21", f"${ind['ema21']:,.0f}")
    with i9:  st.metric("EMA-50", f"${ind['ema50']:,.0f}")
    with i10: st.metric("Vol Ratio", f"{ind['vol_ratio']:.2f}x")
    with i11: st.metric("OBV Slope", f"{ind['obv_slope']:+,.0f}")
    with i12: st.metric("BB Upper",  f"${ind['bb_upper']:,.0f}")

# ─────────────────────────────────────────────────────────────────
#  ORDER BOOK DEPTH CHART
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
ob_col, sent_col = st.columns(2)

with ob_col:
    st.markdown("#### 📖 Order Book Depth")
    bids = ob.get("bids", [])[:25]
    asks = ob.get("asks", [])[:25]
    if bids and asks:
        bid_prices = [p for p, _ in bids]
        bid_vols   = [q for _, q in bids]
        ask_prices = [p for p, _ in asks]
        ask_vols   = [q for _, q in asks]

        # Cumulative
        bid_cum = np.cumsum(bid_vols)[::-1]
        ask_cum = np.cumsum(ask_vols)

        fig_ob = go.Figure()
        fig_ob.add_trace(go.Scatter(
            x=bid_prices[::-1], y=bid_cum,
            fill="tozeroy", name="Bids",
            line=dict(color="#00e676", width=1.5),
            fillcolor="rgba(0,230,118,0.15)"
        ))
        fig_ob.add_trace(go.Scatter(
            x=ask_prices, y=ask_cum,
            fill="tozeroy", name="Asks",
            line=dict(color="#ff5252", width=1.5),
            fillcolor="rgba(255,82,82,0.15)"
        ))
        fig_ob.update_layout(
            height=260, plot_bgcolor="#080c14", paper_bgcolor="#05080f",
            margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="Space Mono, monospace", size=9, color="#5a7a8c"),
            legend=dict(orientation="h", font_size=9),
        )
        fig_ob.update_xaxes(gridcolor="#0f1825", zeroline=False)
        fig_ob.update_yaxes(gridcolor="#0f1825", zeroline=False)
        st.plotly_chart(fig_ob, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px;">
          <div class="sig-card"><div class="sig-title">Bid Volume</div><div class="sig-value bull">{ob_data['bid_vol']:.1f} BTC</div></div>
          <div class="sig-card"><div class="sig-title">Ask Volume</div><div class="sig-value bear">{ob_data['ask_vol']:.1f} BTC</div></div>
          <div class="sig-card"><div class="sig-title">Imbalance</div><div class="sig-value {'bull' if ob_data['imbalance']>0 else 'bear'}">{ob_data['imbalance']:+.1f}%</div></div>
        </div>
        """, unsafe_allow_html=True)

with sent_col:
    st.markdown("#### 🧠 Sentiment")
    # Fear & Greed gauge
    fg_val = fg["value"]
    fg_color = (
        "#ff5252" if fg_val < 25 else
        "#ff8c00" if fg_val < 45 else
        "#ffeb3b" if fg_val < 55 else
        "#69f0ae" if fg_val < 75 else
        "#00e676"
    )
    fig_fg = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fg_val,
        number={"font": {"color": fg_color, "family": "Space Mono", "size": 32}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#2a3a4c", "tickfont": {"size": 9}},
            "bar":   {"color": fg_color, "thickness": 0.25},
            "bgcolor": "#080c14",
            "bordercolor": "#0f1825",
            "steps": [
                {"range": [0,  25], "color": "rgba(255,82,82,0.15)"},
                {"range": [25, 45], "color": "rgba(255,140,0,0.1)"},
                {"range": [45, 55], "color": "rgba(255,235,59,0.08)"},
                {"range": [55, 75], "color": "rgba(105,240,174,0.1)"},
                {"range": [75,100], "color": "rgba(0,230,118,0.15)"},
            ],
        },
        title={"text": f"Fear & Greed: {fg['label']}", "font": {"color": "#5a7a8c", "size": 11, "family": "Space Mono"}},
    ))
    fig_fg.update_layout(
        height=200, paper_bgcolor="#05080f",
        margin=dict(l=20, r=20, t=30, b=0),
        font=dict(color="#5a7a8c")
    )
    st.plotly_chart(fig_fg, use_container_width=True, config={"displayModeBar": False})

    # News feed
    st.markdown("**Latest BTC News**")
    if news:
        for n in news[:5]:
            s_col = "#00e676" if n["sentiment"]=="bull" else ("#ff5252" if n["sentiment"]=="bear" else "#ff8c00")
            icon  = "▲" if n["sentiment"]=="bull" else ("▼" if n["sentiment"]=="bear" else "◆")
            st.markdown(f"""
            <div class="news-item">
              <div class="news-title">{n['title'][:90]}{'...' if len(n['title'])>90 else ''}</div>
              <div class="news-meta" style="color:{s_col}">{icon} {n['sentiment'].upper()} &nbsp;·&nbsp; 👍{n['bull']} 👎{n['bear']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:11px;color:#2a3a4c;">No news data available</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  POLYMARKET BTC MARKETS
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🎯 Active Polymarket BTC Markets")

pred_dir = prediction["cls"]  # bull / bear / neut

if pm_mkts:
    for m in pm_mkts:
        question   = m.get("question") or m.get("title") or "Unknown"
        cid        = m.get("conditionId") or m.get("id") or ""
        end_raw    = m.get("endDate") or m.get("end_date") or m.get("resolveDate")
        tokens     = m.get("tokens", []) or m.get("outcomes", [])
        yes_p, no_p = None, None
        for t in tokens:
            if not isinstance(t, dict): continue
            out = (t.get("outcome") or "").upper()
            p   = t.get("price") or t.get("lastTradePrice")
            if p:
                if "YES" in out: yes_p = float(p)
                if "NO"  in out: no_p  = float(p)

        # Alignment with our prediction
        q_lower = question.lower()
        is_up   = any(w in q_lower for w in ["above","higher","exceed","over","up","bull","rise"])
        is_down = any(w in q_lower for w in ["below","lower","under","fall","drop","bear","crash"])

        if (pred_dir == "bull" and is_up) or (pred_dir == "bear" and is_down):
            align_cls  = "pm-align-bull"
            align_text = "✦ ALIGNED WITH SIGNAL"
        elif (pred_dir == "bull" and is_down) or (pred_dir == "bear" and is_up):
            align_cls  = "pm-align-bear"
            align_text = "✗ AGAINST SIGNAL"
        else:
            align_cls  = "pm-align-neut"
            align_text = "◆ NEUTRAL"

        hours_left = ""
        if end_raw:
            try:
                edt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
                hl  = (edt - datetime.now(timezone.utc)).total_seconds() / 3600
                hours_left = f"{hl:.1f}h"
            except Exception:
                pass

        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-question">{question}</div>
          <div class="pm-grid">
            <div>
              <div class="pm-metric-label">YES Price</div>
              <div class="pm-metric-value">{f'{yes_p:.3f}' if yes_p else '—'}</div>
            </div>
            <div>
              <div class="pm-metric-label">NO Price</div>
              <div class="pm-metric-value">{f'{no_p:.3f}' if no_p else '—'}</div>
            </div>
            <div>
              <div class="pm-metric-label">Closes In</div>
              <div class="pm-metric-value">{hours_left or '—'}</div>
            </div>
          </div>
          <div class="pm-metric-label {align_cls}" style="font-size:10px;font-weight:700;">{align_text}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            if yes_p and (1 - yes_p) / yes_p > 0.05:
                profit_pct = round(((1 - yes_p) / yes_p) * 100, 1)
                st.markdown(
                    f'<span style="font-family:Space Mono,monospace;font-size:10px;color:#00e676;">YES profit if correct: +{profit_pct}%</span>',
                    unsafe_allow_html=True
                )
        with c2:
            if cid:
                st.markdown(
                    f'<a href="https://polymarket.com/event/{cid}" target="_blank" style="font-family:Space Mono,monospace;font-size:10px;color:#00c9ff;text-decoration:none;">↗ Open Market</a>',
                    unsafe_allow_html=True
                )
else:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:12px;color:#2a3a4c;padding:20px;">No active BTC markets found on Polymarket right now.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  AUTO REFRESH
# ─────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
