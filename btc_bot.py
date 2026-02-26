"""
PolyCopy — BTC 5-Minute Prediction Bot
Targets: https://polymarket.com/event/btc-updown-5m-{timestamp}
Slug is deterministic — calculated from current UTC time.
Data: Kraken API (candles, ticker, orderbook) + Fear & Greed + News
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
import math

st.set_page_config(page_title="BTC 5-Min Bot", page_icon="₿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#05080f;}
section[data-testid="stSidebar"]{background:#080c14!important;border-right:1px solid #0f1825;}
section[data-testid="stSidebar"] *{color:#8a9bb0!important;}
[data-testid="metric-container"]{background:#080c14;border:1px solid #0f1825;border-radius:10px;padding:14px!important;}
[data-testid="metric-container"] label{font-family:'Space Mono',monospace!important;font-size:8px!important;text-transform:uppercase;letter-spacing:1.5px;color:#2a3a4c!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Space Mono',monospace!important;font-size:20px!important;color:#f0f8ff!important;}
.stButton>button{background:linear-gradient(135deg,#ff8c00,#ff6b00)!important;color:#000!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:800!important;text-transform:uppercase!important;padding:10px 24px!important;}
.big-signal{border-radius:16px;padding:28px 20px;text-align:center;margin-bottom:16px;}
.big-signal.up{background:rgba(0,230,118,.07);border:2px solid rgba(0,230,118,.4);}
.big-signal.down{background:rgba(255,82,82,.07);border:2px solid rgba(255,82,82,.4);}
.big-signal.wait{background:rgba(255,140,0,.07);border:2px solid rgba(255,140,0,.4);}
.sig-arrow{font-size:52px;font-weight:800;line-height:1;margin-bottom:4px;}
.sig-arrow.up{color:#00e676;}.sig-arrow.down{color:#ff5252;}.sig-arrow.wait{color:#ff8c00;}
.sig-word{font-family:'Space Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:3px;color:#5a7a8c;margin-bottom:12px;}
.sig-conf{font-family:'Space Mono',monospace;font-size:34px;font-weight:700;}
.sig-conf.up{color:#00e676;}.sig-conf.down{color:#ff5252;}.sig-conf.wait{color:#ff8c00;}
.sig-sub{font-family:'Space Mono',monospace;font-size:9px;color:#2a3a4c;margin-top:4px;}
.ind-row{display:flex;justify-content:space-between;align-items:center;padding:7px 12px;border-bottom:1px solid #0f1825;}
.ind-row:last-child{border-bottom:none;}
.ind-name{font-family:'Space Mono',monospace;font-size:9px;color:#5a7a8c;text-transform:uppercase;}
.ind-val{font-family:'Space Mono',monospace;font-size:10px;font-weight:700;}
.ind-val.up{color:#00e676;}.ind-val.down{color:#ff5252;}.ind-val.neut{color:#ff8c00;}
.pm-live{background:linear-gradient(135deg,rgba(0,230,118,.08),rgba(0,201,255,.05));border:2px solid rgba(0,230,118,.3);border-radius:12px;padding:18px 20px;margin-bottom:12px;}
.pm-live-title{font-family:'Space Mono',monospace;font-size:9px;color:#00e676;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}
.pm-live-q{font-size:15px;font-weight:700;color:#f0f8ff;line-height:1.4;margin-bottom:14px;}
.pm-prices{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.pm-price-box{background:#0a0e18;border:1px solid #0f1825;border-radius:8px;padding:10px 14px;}
.pm-price-label{font-family:'Space Mono',monospace;font-size:8px;color:#2a3a4c;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;}
.pm-price-val{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;}
.pm-price-val.up{color:#00e676;}.pm-price-val.down{color:#ff5252;}
.pm-price-profit{font-family:'Space Mono',monospace;font-size:10px;margin-top:3px;}
.pm-price-profit.up{color:#00e676;}.pm-price-profit.down{color:#ff5252;}
.timer-big{font-family:'Space Mono',monospace;font-size:32px;font-weight:700;color:#00c9ff;text-align:center;}
.timer-label{font-family:'Space Mono',monospace;font-size:9px;color:#2a3a4c;text-transform:uppercase;letter-spacing:1.5px;text-align:center;margin-bottom:4px;}
.rec-box{border-radius:8px;padding:10px 14px;text-align:center;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.rec-box.up{background:rgba(0,230,118,.12);border:1px solid rgba(0,230,118,.3);color:#00e676;}
.rec-box.down{background:rgba(255,82,82,.12);border:1px solid rgba(255,82,82,.3);color:#ff5252;}
.rec-box.wait{background:rgba(255,140,0,.1);border:1px solid rgba(255,140,0,.3);color:#ff8c00;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MARKET TIMING  — deterministic slug calculation
# ─────────────────────────────────────────────
def get_current_market_info():
    """
    The Polymarket BTC 5m slug is btc-updown-5m-{ts} where ts is the
    Unix timestamp of the END of the current 5-minute window, rounded
    to the nearest 5-minute boundary (300s intervals).
    """
    now_ts    = int(datetime.now(timezone.utc).timestamp())
    # Round UP to next 5-min boundary
    end_ts    = math.ceil(now_ts / 300) * 300
    start_ts  = end_ts - 300
    secs_left = end_ts - now_ts
    slug      = f"btc-updown-5m-{end_ts}"
    url       = f"https://polymarket.com/event/{slug}"
    return {
        "slug":      slug,
        "url":       url,
        "end_ts":    end_ts,
        "start_ts":  start_ts,
        "secs_left": secs_left,
        "end_dt":    datetime.fromtimestamp(end_ts, tz=timezone.utc),
    }

# ─────────────────────────────────────────────
#  API CALLS
# ─────────────────────────────────────────────
GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
KRAKEN= "https://api.kraken.com/0/public"
HDR   = {"Accept": "application/json"}

@st.cache_data(ttl=25)
def fetch_candles(limit=200):
    try:
        r = requests.get(f"{KRAKEN}/OHLC", params={"pair":"XBTUSD","interval":5},
                         headers=HDR, timeout=12)
        data   = r.json()
        result = data["result"]
        key    = [k for k in result if k != "last"][0]
        raw    = result[key][-limit:]
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vwap","volume","trades"])
        for c in ["open","high","low","close","vwap","volume"]: df[c] = pd.to_numeric(df[c])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df["taker_buy_base"] = df["volume"] * 0.52
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Candle error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_ticker():
    try:
        r      = requests.get(f"{KRAKEN}/Ticker", params={"pair":"XBTUSD"}, headers=HDR, timeout=8)
        result = r.json()["result"]
        t      = result[list(result.keys())[0]]
        price  = float(t["c"][0]); open_ = float(t["o"])
        return {"price":price,"pct_chg":round((price-open_)/open_*100,2),
                "high":float(t["h"][1]),"low":float(t["l"][1]),"vol":float(t["v"][1])}
    except: return {}

@st.cache_data(ttl=10)
def fetch_orderbook():
    try:
        r      = requests.get(f"{KRAKEN}/Depth", params={"pair":"XBTUSD","count":50},
                               headers=HDR, timeout=8)
        result = r.json()["result"]
        book   = result[list(result.keys())[0]]
        bids   = [(float(p),float(q)) for p,q,_ in book.get("bids",[])]
        asks   = [(float(p),float(q)) for p,q,_ in book.get("asks",[])]
        return {"bids":bids,"asks":asks}
    except: return {"bids":[],"asks":[]}

@st.cache_data(ttl=55)   # refresh just before each new 5m window
def fetch_pm_market(slug: str):
    """Fetch the specific 5m BTC market by its deterministic slug."""
    try:
        r    = requests.get(f"{GAMMA}/markets", params={"slug": slug}, headers=HDR, timeout=12)
        data = r.json()
        mkts = data if isinstance(data, list) else data.get("markets", [])
        if mkts:
            return mkts[0]
        # Fallback: search by tag/keyword
        r2   = requests.get(f"{GAMMA}/markets",
               params={"active":"true","closed":"false","tag":"crypto","limit":200},
               headers=HDR, timeout=12)
        data2 = r2.json()
        all_m = data2 if isinstance(data2, list) else data2.get("markets", [])
        for m in all_m:
            if slug in (m.get("slug") or ""):
                return m
        return None
    except: return None

@st.cache_data(ttl=10)
def fetch_clob_prices(market):
    """Fetch live YES/NO prices from the CLOB order book."""
    if not market: return {"yes": None, "no": None}
    tokens = market.get("tokens", []) or []
    prices = {"yes": None, "no": None}
    for tok in tokens:
        if not isinstance(tok, dict): continue
        tid    = tok.get("token_id") or tok.get("tokenId") or ""
        outcome= (tok.get("outcome") or "").upper()
        if not tid: continue
        try:
            r = requests.get(f"{CLOB}/price",
                             params={"token_id": tid, "side": "buy"},
                             headers=HDR, timeout=6)
            p = float(r.json().get("price", 0))
            if "YES" in outcome or "UP" in outcome:   prices["yes"] = p
            elif "NO" in outcome or "DOWN" in outcome: prices["no"]  = p
        except: pass
    # Fallback from market metadata
    for tok in tokens:
        if not isinstance(tok, dict): continue
        outcome = (tok.get("outcome") or "").upper()
        p = tok.get("price") or tok.get("lastTradePrice")
        if p:
            if ("YES" in outcome or "UP" in outcome) and prices["yes"] is None:
                prices["yes"] = float(p)
            if ("NO" in outcome or "DOWN" in outcome) and prices["no"] is None:
                prices["no"] = float(p)
    return prices

@st.cache_data(ttl=300)
def fetch_fear_greed():
    try:
        data = requests.get("https://api.alternative.me/fng/?limit=2",timeout=8).json().get("data",[{}])
        return {"value":int(data[0].get("value",50)),"label":data[0].get("value_classification","Neutral"),
                "prev":int(data[1].get("value",50)) if len(data)>1 else 50}
    except: return {"value":50,"label":"Neutral","prev":50}

@st.cache_data(ttl=120)
def fetch_news():
    try:
        r = requests.get("https://cryptopanic.com/api/v1/posts/",
            params={"auth_token":"public","currencies":"BTC","kind":"news","public":"true"},timeout=10)
        items=[]
        for item in r.json().get("results",[])[:6]:
            v=item.get("votes",{});b,be=v.get("positive",0),v.get("negative",0)
            items.append({"title":item.get("title",""),"bull":b,"bear":be,
                          "sent":"up" if b>be else("down" if be>b else "neut")})
        return items
    except: return []

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def compute_indicators(df):
    if df.empty or len(df)<60: return {}
    c=df["close"].values;h=df["high"].values;l=df["low"].values;v=df["volume"].values
    def ema(src,n): return pd.Series(src).ewm(span=n,adjust=False).mean().values
    d=np.diff(c);gain=np.where(d>0,d,0.);loss=np.where(d<0,-d,0.)
    ag=pd.Series(gain).ewm(span=14,adjust=False).mean().values
    al=pd.Series(loss).ewm(span=14,adjust=False).mean().values
    rsi_s=100-(100/(1+np.where(al==0,100,ag/(al+1e-9))))
    rsi_full=np.concatenate([[np.nan],rsi_s])
    fast=ema(c,12);slow=ema(c,26);macd_l=fast-slow;sig_l=ema(macd_l,9);hist_l=macd_l-sig_l
    sma20=pd.Series(c).rolling(20).mean().values;std20=pd.Series(c).rolling(20).std().values
    bb_up=sma20+2*std20;bb_dn=sma20-2*std20
    bb_pos=float((c[-1]-bb_dn[-1])/(bb_up[-1]-bb_dn[-1]+1e-9))
    lo14=pd.Series(l).rolling(14).min().values;hi14=pd.Series(h).rolling(14).max().values
    stk=100*(c-lo14)/(hi14-lo14+1e-9);std_=pd.Series(stk).rolling(3).mean().values
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    atr=pd.Series(tr).ewm(span=14,adjust=False).mean().values
    tp=(h+l+c)/3;vwap=(tp*v).cumsum()/(v.cumsum()+1e-9)
    sign=np.sign(np.diff(c));obv=np.cumsum(np.concatenate([[0],sign*v[1:]]))
    buy_v=df["taker_buy_base"].values;sell_v=v-buy_v
    vol_ratio=float(buy_v[-5:].mean()/(sell_v[-5:].mean()+1e-9))
    e9=ema(c,9);e21=ema(c,21);e50=ema(c,50)
    return {
        "close":float(c[-1]),"prev_close":float(c[-2]),
        "rsi":float(rsi_full[-1]),"rsi_s":rsi_full,
        "macd":float(macd_l[-1]),"macd_sig":float(sig_l[-1]),"macd_hist":float(hist_l[-1]),
        "macd_s":macd_l,"signal_s":sig_l,"hist_s":hist_l,
        "bb_upper":float(bb_up[-1]),"bb_lower":float(bb_dn[-1]),
        "bb_mid":float(sma20[-1]),"bb_pos":bb_pos,"bb_up_s":bb_up,"bb_dn_s":bb_dn,
        "stoch_k":float(stk[-1]),"stoch_d":float(std_[-1]),
        "atr":float(atr[-1]) if not np.isnan(atr[-1]) else 0,
        "vwap":float(vwap[-1]),
        "obv_slope":float(np.polyfit(range(5),obv[-5:],1)[0]),
        "vol_ratio":vol_ratio,
        "ema9":float(e9[-1]),"ema21":float(e21[-1]),"ema50":float(e50[-1]),
        "ema9_s":e9,"ema21_s":e21,"ema50_s":e50,"df":df,
    }

def analyze_ob(ob):
    bids=ob.get("bids",[]);asks=ob.get("asks",[])
    if not bids or not asks: return {"imbalance":0.,"signal":"neut","bid_vol":0,"ask_vol":0}
    bv=sum(q for _,q in bids);av=sum(q for _,q in asks)
    imb=(bv-av)/(bv+av+1e-9)*100
    return {"imbalance":round(imb,1),"bid_vol":round(bv,2),"ask_vol":round(av,2),
            "signal":"up" if imb>8 else("down" if imb<-8 else "neut")}

def analyze_sent(news,fg):
    bull=sum(1 for n in news if n["sent"]=="up");bear=sum(1 for n in news if n["sent"]=="down")
    ns=(bull-bear)/(len(news) or 1);fgs=(fg.get("value",50)-50)/50;comb=ns*0.4+fgs*0.6
    return {"fg_value":fg.get("value",50),"fg_label":fg.get("label","Neutral"),
            "signal":"up" if comb>0.1 else("down" if comb<-0.1 else "neut")}

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def predict(ind, ob, sent):
    if not ind:
        return {"direction":"WAIT","cls":"wait","confidence":0,
                "signals":[],"up_score":0,"down_score":0}
    sigs=[]
    def s(name,d,st,r): sigs.append({"name":name,"dir":d,"strength":st,"reason":r})

    rsi=ind["rsi"]
    if   rsi<30: s("RSI","up",  88,f"Oversold {rsi:.0f}")
    elif rsi>70: s("RSI","down",88,f"Overbought {rsi:.0f}")
    elif rsi<45: s("RSI","up",  55,f"Below mid {rsi:.0f}")
    elif rsi>55: s("RSI","down",55,f"Above mid {rsi:.0f}")
    else:        s("RSI","neut",0, f"Neutral {rsi:.0f}")

    hh=ind["macd_hist"]
    if   hh>0 and ind["macd"]>ind["macd_sig"]: s("MACD","up",  72,"Bull crossover")
    elif hh<0 and ind["macd"]<ind["macd_sig"]: s("MACD","down",72,"Bear crossover")
    elif hh>0: s("MACD","up",  45,"Pos histogram")
    else:      s("MACD","down",45,"Neg histogram")

    bp=ind["bb_pos"]
    if   bp<0.1: s("BB","up",  78,"Near lower band")
    elif bp>0.9: s("BB","down",78,"Near upper band")
    elif bp>0.5: s("BB","up",  38,"Above midline")
    else:        s("BB","down",38,"Below midline")

    e9,e21,e50=ind["ema9"],ind["ema21"],ind["ema50"]
    if   e9>e21>e50: s("EMA","up",  82,"Bull stack 9>21>50")
    elif e9<e21<e50: s("EMA","down",82,"Bear stack 9<21<50")
    elif e9>e21:     s("EMA","up",  48,"Short bull")
    else:            s("EMA","down",48,"Short bear")

    sk=ind["stoch_k"]
    if   sk<20: s("Stoch","up",  80,f"Oversold {sk:.0f}")
    elif sk>80: s("Stoch","down",80,f"Overbought {sk:.0f}")
    elif sk>ind["stoch_d"]: s("Stoch","up",  42,"K>D")
    else:                   s("Stoch","down",42,"K<D")

    cl=ind["close"]
    if   cl>ind["vwap"]*1.001: s("VWAP","up",  65,"Above VWAP")
    elif cl<ind["vwap"]*0.999: s("VWAP","down",65,"Below VWAP")
    else:                       s("VWAP","neut",0, "At VWAP")

    if ind["obv_slope"]>0: s("OBV","up",  58,"Rising OBV")
    else:                   s("OBV","down",58,"Falling OBV")

    vr=ind["vol_ratio"]
    if   vr>1.3: s("Vol","up",  70,f"Buy pres {vr:.1f}x")
    elif vr<0.7: s("Vol","down",70,f"Sell pres {1/vr:.1f}x")
    else:        s("Vol","neut",0, "Balanced")

    imb=ob.get("imbalance",0)
    if   imb>8:  s("OB","up",  72,f"Bid-heavy +{imb:.0f}%")
    elif imb<-8: s("OB","down",72,f"Ask-heavy {imb:.0f}%")
    else:        s("OB","neut",0, f"Balanced {imb:.0f}%")

    fg=sent.get("fg_value",50)
    if   fg<25:                  s("Sent","up",  64,"Extreme Fear")
    elif fg>75:                  s("Sent","down",64,"Extreme Greed")
    elif sent["signal"]=="up":   s("Sent","up",  42,"Bull news")
    elif sent["signal"]=="down": s("Sent","down",42,"Bear news")
    else:                        s("Sent","neut",0, "Neutral")

    up_sc=sum(x["strength"] for x in sigs if x["dir"]=="up")
    dn_sc=sum(x["strength"] for x in sigs if x["dir"]=="down")
    total=up_sc+dn_sc+1e-9
    if   up_sc>dn_sc: return {"direction":"UP",  "cls":"up",  "confidence":round(up_sc/total*100),"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}
    elif dn_sc>up_sc: return {"direction":"DOWN","cls":"down","confidence":round(dn_sc/total*100),"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}
    else:             return {"direction":"WAIT","cls":"wait","confidence":50,"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}

# ─────────────────────────────────────────────
#  CHART  — two-trace method for bar colors (works on plotly + Python 3.13)
# ─────────────────────────────────────────────
def build_chart(ind):
    df=ind.get("df")
    if df is None or df.empty: return go.Figure()
    ts=df["ts"]
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,
                       row_heights=[0.50,0.16,0.17,0.17],vertical_spacing=0.02)

    # Candles
    fig.add_trace(go.Candlestick(x=ts,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="BTC",increasing_fillcolor="#00e676",increasing_line_color="#00e676",
        decreasing_fillcolor="#ff5252",decreasing_line_color="#ff5252",
        line=dict(width=1),showlegend=False),row=1,col=1)

    # EMAs
    for vals,color,nm in [(ind["ema9_s"],"#ff8c00","EMA9"),(ind["ema21_s"],"#00c9ff","EMA21"),(ind["ema50_s"],"#b967ff","EMA50")]:
        fig.add_trace(go.Scatter(x=ts,y=vals,name=nm,line=dict(color=color,width=1.2)),row=1,col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=ts,y=ind["bb_up_s"],name="BB",showlegend=False,
        line=dict(color="rgba(100,150,255,0.35)",width=1,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["bb_dn_s"],name="BB",fill="tonexty",showlegend=False,
        fillcolor="rgba(100,150,255,0.04)",line=dict(color="rgba(100,150,255,0.35)",width=1,dash="dot")),row=1,col=1)

    # Volume — split into two boolean-indexed traces (avoids rgba list bug on Python 3.13)
    up_mask = (df["close"] >= df["open"]).values
    ts_arr  = np.array(ts)
    vol_arr = df["volume"].values
    fig.add_trace(go.Bar(x=ts_arr[up_mask],  y=vol_arr[up_mask],  name="Vol",
                          marker_color="#00e676", opacity=0.45, showlegend=False),row=2,col=1)
    fig.add_trace(go.Bar(x=ts_arr[~up_mask], y=vol_arr[~up_mask], name="Vol",
                          marker_color="#ff5252", opacity=0.45, showlegend=False),row=2,col=1)

    # MACD histogram — same two-trace approach
    hist_arr = ind["hist_s"]
    pos_mask = hist_arr >= 0
    fig.add_trace(go.Bar(x=ts_arr[pos_mask],  y=hist_arr[pos_mask],  name="MACD",
                          marker_color="#00e676", opacity=0.65, showlegend=False),row=3,col=1)
    fig.add_trace(go.Bar(x=ts_arr[~pos_mask], y=hist_arr[~pos_mask], name="MACD",
                          marker_color="#ff5252", opacity=0.65, showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["macd_s"],  name="MACD", line=dict(color="#ff8c00",width=1.2)),row=3,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["signal_s"],name="Sig",  line=dict(color="#00c9ff",width=1.2)),row=3,col=1)

    # RSI
    fig.add_trace(go.Scatter(x=ts,y=ind["rsi_s"],name="RSI",line=dict(color="#b967ff",width=1.5)),row=4,col=1)
    for lvl,col in [(70,"rgba(255,82,82,0.3)"),(30,"rgba(0,230,118,0.3)"),(50,"rgba(255,255,255,0.08)")]:
        fig.add_hline(y=lvl,line_dash="dot",line_color=col,row=4,col=1)

    fig.update_layout(height=500,plot_bgcolor="#080c14",paper_bgcolor="#05080f",
        font=dict(family="Space Mono,monospace",size=9,color="#5a7a8c"),
        margin=dict(l=0,r=0,t=8,b=0),xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,font_size=9,bgcolor="rgba(0,0,0,0)"),
        barmode="overlay")
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#0f1825",zeroline=False,showticklabels=(i==4),row=i,col=1)
        fig.update_yaxes(gridcolor="#0f1825",zeroline=False,row=i,col=1)
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ₿ BTC 5-Min Bot")
    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    candle_limit = st.slider("Candles", 100, 500, 200, 50)
    st.markdown("---")
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:#2a3a4c;line-height:2;">⚠ NOT FINANCIAL ADVICE<br>Educational use only.<br>Never bet more than<br>you can afford to lose.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER + TIMING
# ─────────────────────────────────────────────
mkt_info = get_current_market_info()
secs_left = mkt_info["secs_left"]
mins_l    = secs_left // 60
secs_l    = secs_left % 60

st.markdown(f"""
<div style='font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#f0f8ff;'>
  ₿TC <span style='color:#ff8c00;'>5-Minute</span> Prediction Bot
</div>
<div style='font-family:Space Mono,monospace;font-size:9px;color:#2a3a4c;text-transform:uppercase;
     letter-spacing:2px;margin:3px 0 16px;'>
  Targeting &nbsp;·&nbsp;
  <a href='{mkt_info["url"]}' target='_blank' style='color:#00c9ff;text-decoration:none;'>
    {mkt_info["slug"]}
  </a>
</div>
""", unsafe_allow_html=True)

col_btn,_=st.columns([1,7])
with col_btn: st.button("⟳ Refresh", use_container_width=True)
st.markdown("---")

# ─────────────────────────────────────────────
#  FETCH DATA
# ─────────────────────────────────────────────
with st.spinner("Fetching live data..."):
    df_5m   = fetch_candles(candle_limit)
    ticker  = fetch_ticker()
    ob      = fetch_orderbook()
    fg      = fetch_fear_greed()
    news    = fetch_news()
    pm_mkt  = fetch_pm_market(mkt_info["slug"])
    clob_px = fetch_clob_prices(pm_mkt)

ind  = compute_indicators(df_5m)
oba  = analyze_ob(ob)
sent = analyze_sent(news, fg)
pred = predict(ind, oba, sent)

price   = ticker.get("price",0)
pct_chg = ticker.get("pct_chg",0)

# ─────────────────────────────────────────────
#  PRICE METRICS
# ─────────────────────────────────────────────
m1,m2,m3,m4,m5=st.columns(5)
with m1: st.metric("BTC/USD",    f"${price:,.2f}", f"{pct_chg:+.2f}%")
with m2: st.metric("24h High",   f"${ticker.get('high',0):,.0f}")
with m3: st.metric("24h Low",    f"${ticker.get('low',0):,.0f}")
with m4: st.metric("24h Volume", f"{ticker.get('vol',0):,.1f} BTC")
with m5: st.metric("Fear & Greed",f"{fg['value']} — {fg['label']}", f"{fg['value']-fg['prev']:+d}")
st.markdown("---")

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    st.markdown("#### 📈 5m BTC/USD · Kraken")
    if ind:
        st.plotly_chart(build_chart(ind), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.warning("Not enough candle data (need 60+).")

with right:
    # ── Countdown ──
    st.markdown(f"""
    <div style='background:#080c14;border:1px solid #0f1825;border-radius:10px;padding:14px;text-align:center;margin-bottom:12px;'>
      <div class='timer-label'>This window closes in</div>
      <div class='timer-big'>{mins_l}:{secs_l:02d}</div>
      <div style='font-family:Space Mono,monospace;font-size:8px;color:#2a3a4c;margin-top:4px;'>
        {mkt_info["end_dt"].strftime("%H:%M UTC")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal ──
    st.markdown("#### ⚡ Signal")
    cls   = pred["cls"]
    arrow = "▲" if cls=="up" else ("▼" if cls=="down" else "◆")
    st.markdown(f"""
    <div class='big-signal {cls}'>
      <div class='sig-arrow {cls}'>{arrow}</div>
      <div class='sig-word'>{pred['direction']}</div>
      <div class='sig-conf {cls}'>{pred['confidence']}%</div>
      <div class='sig-sub'>confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score bars ──
    up_w=min(pred.get("up_score",0)/7,100); dn_w=min(pred.get("down_score",0)/7,100)
    st.markdown(f"""
    <div style='margin-bottom:12px;'>
      <div class='ind-row'><span class='ind-name'>UP</span>
        <div style='flex:1;margin:0 8px;height:4px;background:#0f1825;border-radius:2px;overflow:hidden;'>
          <div style='width:{up_w}%;height:100%;background:#00e676;'></div></div>
        <span class='ind-val up'>{pred.get("up_score",0)}</span></div>
      <div class='ind-row'><span class='ind-name'>DOWN</span>
        <div style='flex:1;margin:0 8px;height:4px;background:#0f1825;border-radius:2px;overflow:hidden;'>
          <div style='width:{dn_w}%;height:100%;background:#ff5252;'></div></div>
        <span class='ind-val down'>{pred.get("down_score",0)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal breakdown ──
    rows=""
    for sig in pred["signals"]:
        d=sig["dir"];icon="▲" if d=="up" else("▼" if d=="down" else "◆")
        c2="up" if d=="up" else("down" if d=="down" else "neut")
        rows+=f"<div class='ind-row'><span class='ind-name'>{sig['name']}</span><span class='ind-val {c2}'>{icon} {sig['reason']}</span></div>"
    st.markdown(f"<div style='background:#080c14;border:1px solid #0f1825;border-radius:10px;margin-bottom:12px;'>{rows}</div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIVE POLYMARKET MARKET BOX
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🎯 Live Polymarket Market")

yes_p = clob_px.get("yes")
no_p  = clob_px.get("no")
q     = (pm_mkt.get("question") or pm_mkt.get("title") or mkt_info["slug"]) if pm_mkt else mkt_info["slug"]

yes_profit = f"+{round(((1-yes_p)/yes_p)*100,1)}%" if yes_p and 0.01<yes_p<0.99 else "—"
no_profit  = f"+{round(((1-no_p)/no_p)*100,1)}%"  if no_p  and 0.01<no_p <0.99 else "—"

# Which side does our prediction recommend?
if pred["cls"] == "up":
    rec_side = "BUY YES (UP)";  rec_cls = "up"
elif pred["cls"] == "down":
    rec_side = "BUY NO (DOWN)"; rec_cls = "down"
else:
    rec_side = "WAIT — low confidence"; rec_cls = "wait"

st.markdown(f"""
<div class='pm-live'>
  <div class='pm-live-title'>● LIVE MARKET</div>
  <div class='pm-live-q'>{q}</div>
  <div class='pm-prices'>
    <div class='pm-price-box'>
      <div class='pm-price-label'>YES (BTC UP)</div>
      <div class='pm-price-val up'>{f'{yes_p:.3f}' if yes_p else '—'}</div>
      <div class='pm-price-profit up'>{yes_profit} if correct</div>
    </div>
    <div class='pm-price-box'>
      <div class='pm-price-label'>NO (BTC DOWN)</div>
      <div class='pm-price-val down'>{f'{no_p:.3f}' if no_p else '—'}</div>
      <div class='pm-price-profit down'>{no_profit} if correct</div>
    </div>
  </div>
  <div class='rec-box {rec_cls}'>⚡ RECOMMENDATION: {rec_side}</div>
</div>
<a href='{mkt_info["url"]}' target='_blank'
   style='font-family:Space Mono,monospace;font-size:10px;color:#00c9ff;text-decoration:none;'>
  ↗ Open on Polymarket
</a>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INDICATOR VALUES
# ─────────────────────────────────────────────
if ind:
    st.markdown("---")
    st.markdown("#### 📐 Indicator Values")
    i1,i2,i3,i4,i5,i6=st.columns(6)
    with i1: st.metric("RSI-14",      f"{ind['rsi']:.1f}")
    with i2: st.metric("MACD Hist",   f"{ind['macd_hist']:.2f}")
    with i3: st.metric("BB Pos",      f"{ind['bb_pos']*100:.0f}%")
    with i4: st.metric("Stoch %K",    f"{ind['stoch_k']:.1f}")
    with i5: st.metric("VWAP",        f"${ind['vwap']:,.0f}")
    with i6: st.metric("Vol Ratio",   f"{ind['vol_ratio']:.2f}x")

# ─────────────────────────────────────────────
#  ORDER BOOK + SENTIMENT
# ─────────────────────────────────────────────
st.markdown("---")
ob_col,sent_col=st.columns(2)
with ob_col:
    st.markdown("#### 📖 Order Book Depth")
    bids=ob.get("bids",[])[:25];asks=ob.get("asks",[])[:25]
    if bids and asks:
        fig_ob=go.Figure()
        fig_ob.add_trace(go.Scatter(x=[p for p,_ in bids][::-1],
            y=np.cumsum([q for _,q in bids])[::-1],fill="tozeroy",name="Bids",
            line=dict(color="#00e676",width=1.5),fillcolor="rgba(0,230,118,0.1)"))
        fig_ob.add_trace(go.Scatter(x=[p for p,_ in asks],
            y=np.cumsum([q for _,q in asks]),fill="tozeroy",name="Asks",
            line=dict(color="#ff5252",width=1.5),fillcolor="rgba(255,82,82,0.1)"))
        fig_ob.update_layout(height=220,plot_bgcolor="#080c14",paper_bgcolor="#05080f",
            margin=dict(l=0,r=0,t=8,b=0),
            font=dict(family="Space Mono,monospace",size=9,color="#5a7a8c"),
            legend=dict(orientation="h",font_size=9,bgcolor="rgba(0,0,0,0)"))
        fig_ob.update_xaxes(gridcolor="#0f1825",zeroline=False)
        fig_ob.update_yaxes(gridcolor="#0f1825",zeroline=False)
        st.plotly_chart(fig_ob,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<span style="font-family:Space Mono,monospace;font-size:10px;color:{"#00e676" if oba["imbalance"]>0 else "#ff5252"};">Order book imbalance: {oba["imbalance"]:+.1f}% {"(bid-heavy ▲)" if oba["imbalance"]>0 else "(ask-heavy ▼)"}</span>', unsafe_allow_html=True)

with sent_col:
    st.markdown("#### 🧠 Sentiment")
    fg_val=fg["value"]
    fg_color="#ff5252" if fg_val<25 else("#ff8c00" if fg_val<45 else("#ffeb3b" if fg_val<55 else("#69f0ae" if fg_val<75 else "#00e676")))
    fig_fg=go.Figure(go.Indicator(mode="gauge+number",value=fg_val,
        number={"font":{"color":fg_color,"family":"Space Mono","size":28}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#2a3a4c","tickfont":{"size":9}},
               "bar":{"color":fg_color,"thickness":0.25},"bgcolor":"#080c14","bordercolor":"#0f1825",
               "steps":[{"range":[0,25],"color":"rgba(255,82,82,0.1)"},
                        {"range":[25,45],"color":"rgba(255,140,0,0.07)"},
                        {"range":[45,55],"color":"rgba(255,235,59,0.05)"},
                        {"range":[55,75],"color":"rgba(105,240,174,0.07)"},
                        {"range":[75,100],"color":"rgba(0,230,118,0.1)"}]},
        title={"text":f"Fear & Greed · {fg['label']}",
               "font":{"color":"#5a7a8c","size":10,"family":"Space Mono"}}))
    fig_fg.update_layout(height=185,paper_bgcolor="#05080f",margin=dict(l=20,r=20,t=28,b=0))
    st.plotly_chart(fig_fg,use_container_width=True,config={"displayModeBar":False})
    if news:
        for n in news[:4]:
            icon="▲" if n["sent"]=="up" else("▼" if n["sent"]=="down" else "◆")
            c={"up":"#00e676","down":"#ff5252","neut":"#ff8c00"}[n["sent"]]
            st.markdown(f'<div style="background:#080c14;border:1px solid #0f1825;border-radius:7px;padding:8px 12px;margin-bottom:5px;"><div style="font-size:11px;color:#c8d8e8;line-height:1.4;margin-bottom:3px;">{n["title"][:80]}{"..." if len(n["title"])>80 else ""}</div><div style="font-family:Space Mono,monospace;font-size:9px;color:{c};">{icon} {n["sent"].upper()} · 👍{n["bull"]} 👎{n["bear"]}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AUTO REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
