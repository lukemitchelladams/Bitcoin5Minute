"""
PolyCopy — BTC 5-Minute Prediction Bot
Target: Polymarket "Will BTC go UP in the next 5 minutes?" market
Gives UP/DOWN + confidence % before the 1-minute mark each candle.
Data: Kraken (candles, ticker, order book) + Fear & Greed + CryptoPanic news
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time

st.set_page_config(page_title="BTC 5-Min Bot", page_icon="₿", layout="wide", initial_sidebar_state="expanded")

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
.stButton>button{background:linear-gradient(135deg,#ff8c00,#ff6b00)!important;color:#000!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:800!important;text-transform:uppercase!important;letter-spacing:.8px!important;padding:10px 24px!important;}
.big-signal{border-radius:16px;padding:32px 24px;text-align:center;margin-bottom:20px;}
.big-signal.up{background:rgba(0,230,118,.07);border:2px solid rgba(0,230,118,.35);}
.big-signal.down{background:rgba(255,82,82,.07);border:2px solid rgba(255,82,82,.35);}
.big-signal.wait{background:rgba(255,140,0,.07);border:2px solid rgba(255,140,0,.35);}
.signal-arrow{font-size:56px;font-weight:800;line-height:1;margin-bottom:4px;}
.signal-arrow.up{color:#00e676;}.signal-arrow.down{color:#ff5252;}.signal-arrow.wait{color:#ff8c00;}
.signal-word{font-family:'Space Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:3px;color:#5a7a8c;margin-bottom:16px;}
.signal-conf{font-family:'Space Mono',monospace;font-size:36px;font-weight:700;}
.signal-conf.up{color:#00e676;}.signal-conf.down{color:#ff5252;}.signal-conf.wait{color:#ff8c00;}
.signal-sub{font-family:'Space Mono',monospace;font-size:9px;color:#2a3a4c;margin-top:6px;text-transform:uppercase;letter-spacing:1px;}
.ind-row{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f1825;}
.ind-row:last-child{border-bottom:none;}
.ind-name{font-family:'Space Mono',monospace;font-size:10px;color:#5a7a8c;text-transform:uppercase;}
.ind-val{font-family:'Space Mono',monospace;font-size:11px;font-weight:700;}
.ind-val.up{color:#00e676;}.ind-val.down{color:#ff5252;}.ind-val.neut{color:#ff8c00;}
.pm-box{background:#080c14;border:1px solid #0f1825;border-radius:10px;padding:14px 16px;margin-bottom:10px;}
.pm-q{font-size:14px;font-weight:600;color:#f0f8ff;line-height:1.4;margin-bottom:10px;}
.pm-row{display:flex;gap:20px;flex-wrap:wrap;}
.pm-metric{display:flex;flex-direction:column;gap:3px;}
.pm-label{font-family:'Space Mono',monospace;font-size:8px;color:#2a3a4c;text-transform:uppercase;letter-spacing:1px;}
.pm-value{font-family:'Space Mono',monospace;font-size:13px;font-weight:700;color:#f0f8ff;}
.pm-value.up{color:#00e676;}.pm-value.down{color:#ff5252;}.pm-value.neut{color:#ff8c00;}
</style>
""", unsafe_allow_html=True)

KRAKEN  = "https://api.kraken.com/0/public"
HEADERS = {"Accept":"application/json"}

@st.cache_data(ttl=25)
def fetch_candles(limit=200):
    try:
        r = requests.get(f"{KRAKEN}/OHLC", params={"pair":"XBTUSD","interval":5}, headers=HEADERS, timeout=12)
        r.raise_for_status()
        data   = r.json()
        if data.get("error"): return pd.DataFrame()
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
        r      = requests.get(f"{KRAKEN}/Ticker", params={"pair":"XBTUSD"}, headers=HEADERS, timeout=8)
        result = r.json()["result"]
        key    = list(result.keys())[0]
        t      = result[key]
        price  = float(t["c"][0])
        open_  = float(t["o"])
        return {"price":price,"pct_chg":round((price-open_)/open_*100,2),
                "high":float(t["h"][1]),"low":float(t["l"][1]),"vol_24h":float(t["v"][1])}
    except: return {}

@st.cache_data(ttl=10)
def fetch_orderbook():
    try:
        r      = requests.get(f"{KRAKEN}/Depth", params={"pair":"XBTUSD","count":50}, headers=HEADERS, timeout=8)
        result = r.json()["result"]
        key    = list(result.keys())[0]
        book   = result[key]
        bids   = [(float(p),float(q)) for p,q,_ in book.get("bids",[])]
        asks   = [(float(p),float(q)) for p,q,_ in book.get("asks",[])]
        return {"bids":bids,"asks":asks}
    except: return {"bids":[],"asks":[]}

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
            v=item.get("votes",{})
            b,be=v.get("positive",0),v.get("negative",0)
            items.append({"title":item.get("title",""),"bull":b,"bear":be,
                          "sent":"up" if b>be else ("down" if be>b else "neut")})
        return items
    except: return []

@st.cache_data(ttl=60)
def fetch_polymarket_btc():
    try:
        r    = requests.get("https://gamma-api.polymarket.com/markets",
               params={"active":"true","closed":"false","limit":100},headers=HEADERS,timeout=12)
        data = r.json()
        mkts = data if isinstance(data,list) else data.get("markets",[])
        return [m for m in mkts if "bitcoin" in (m.get("question") or m.get("title") or "").lower()
                or "btc" in (m.get("question") or m.get("title") or "").lower()][:8]
    except: return []

def compute_indicators(df):
    if df.empty or len(df)<60: return {}
    c=df["close"].values; h=df["high"].values; l=df["low"].values; v=df["volume"].values
    def ema(src,n): return pd.Series(src).ewm(span=n,adjust=False).mean().values
    d=np.diff(c); gain=np.where(d>0,d,0.); loss=np.where(d<0,-d,0.)
    ag=pd.Series(gain).ewm(span=14,adjust=False).mean().values
    al=pd.Series(loss).ewm(span=14,adjust=False).mean().values
    rsi_s=100-(100/(1+np.where(al==0,100,ag/(al+1e-9))))
    rsi_full=np.concatenate([[np.nan],rsi_s])
    fast=ema(c,12); slow=ema(c,26); macd_l=fast-slow
    sig_l=ema(macd_l,9); hist_l=macd_l-sig_l
    sma20=pd.Series(c).rolling(20).mean().values; std20=pd.Series(c).rolling(20).std().values
    bb_up=sma20+2*std20; bb_dn=sma20-2*std20
    bb_pos=float((c[-1]-bb_dn[-1])/(bb_up[-1]-bb_dn[-1]+1e-9))
    lo14=pd.Series(l).rolling(14).min().values; hi14=pd.Series(h).rolling(14).max().values
    stk=100*(c-lo14)/(hi14-lo14+1e-9); std_=pd.Series(stk).rolling(3).mean().values
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    atr=pd.Series(tr).ewm(span=14,adjust=False).mean().values
    tp=(h+l+c)/3; vwap=(tp*v).cumsum()/(v.cumsum()+1e-9)
    sign=np.sign(np.diff(c)); obv=np.cumsum(np.concatenate([[0],sign*v[1:]]))
    buy_v=df["taker_buy_base"].values; sell_v=v-buy_v
    vol_ratio=float(buy_v[-5:].mean()/(sell_v[-5:].mean()+1e-9))
    e9=ema(c,9); e21=ema(c,21); e50=ema(c,50)
    return {
        "close":float(c[-1]),"prev_close":float(c[-2]),
        "rsi":float(rsi_full[-1]),"rsi_s":rsi_full,
        "macd":float(macd_l[-1]),"macd_sig":float(sig_l[-1]),"macd_hist":float(hist_l[-1]),
        "macd_s":macd_l,"signal_s":sig_l,"hist_s":hist_l,
        "bb_upper":float(bb_up[-1]),"bb_lower":float(bb_dn[-1]),"bb_mid":float(sma20[-1]),"bb_pos":bb_pos,
        "bb_up_s":bb_up,"bb_dn_s":bb_dn,
        "stoch_k":float(stk[-1]),"stoch_d":float(std_[-1]),
        "atr":float(atr[-1]) if not np.isnan(atr[-1]) else 0,
        "vwap":float(vwap[-1]),
        "obv_slope":float(np.polyfit(range(5),obv[-5:],1)[0]),
        "vol_ratio":vol_ratio,
        "ema9":float(e9[-1]),"ema21":float(e21[-1]),"ema50":float(e50[-1]),
        "ema9_s":e9,"ema21_s":e21,"ema50_s":e50,"df":df,
    }

def analyze_orderbook(ob):
    bids=ob.get("bids",[]); asks=ob.get("asks",[])
    if not bids or not asks: return {"imbalance":0.,"bid_vol":0,"ask_vol":0,"signal":"neut"}
    bv=sum(q for _,q in bids); av=sum(q for _,q in asks)
    imb=(bv-av)/(bv+av+1e-9)*100
    return {"imbalance":round(imb,1),"bid_vol":round(bv,2),"ask_vol":round(av,2),
            "signal":"up" if imb>8 else ("down" if imb<-8 else "neut")}

def analyze_sentiment(news,fg):
    bull=sum(1 for n in news if n["sent"]=="up"); bear=sum(1 for n in news if n["sent"]=="down")
    ns=(bull-bear)/(len(news) or 1); fgs=(fg.get("value",50)-50)/50
    comb=ns*0.4+fgs*0.6
    return {"fg_value":fg.get("value",50),"fg_label":fg.get("label","Neutral"),
            "news_bull":bull,"news_bear":bear,
            "signal":"up" if comb>0.1 else ("down" if comb<-0.1 else "neut"),"score":round(comb*100,1)}

def predict(ind,ob,sent):
    if not ind: return {"direction":"WAIT","cls":"wait","confidence":0,"signals":[],"up_score":0,"down_score":0}
    sigs=[]
    def s(name,d,st,r): sigs.append({"name":name,"dir":d,"strength":st,"reason":r})
    rsi=ind["rsi"]
    if   rsi<30: s("RSI","up",  88,f"Oversold ({rsi:.1f})")
    elif rsi>70: s("RSI","down",88,f"Overbought ({rsi:.1f})")
    elif rsi<45: s("RSI","up",  55,f"Below midline ({rsi:.1f})")
    elif rsi>55: s("RSI","down",55,f"Above midline ({rsi:.1f})")
    else:        s("RSI","neut",0, f"Neutral ({rsi:.1f})")
    hh=ind["macd_hist"]
    if   hh>0 and ind["macd"]>ind["macd_sig"]: s("MACD","up",  72,"Bullish crossover")
    elif hh<0 and ind["macd"]<ind["macd_sig"]: s("MACD","down",72,"Bearish crossover")
    elif hh>0: s("MACD","up",  45,"Positive hist")
    else:      s("MACD","down",45,"Negative hist")
    bp=ind["bb_pos"]
    if   bp<0.1: s("Bollinger","up",  78,"Near lower band")
    elif bp>0.9: s("Bollinger","down",78,"Near upper band")
    elif bp>0.5: s("Bollinger","up",  38,"Upper half")
    else:        s("Bollinger","down",38,"Lower half")
    e9,e21,e50=ind["ema9"],ind["ema21"],ind["ema50"]
    if   e9>e21>e50: s("EMA Stack","up",  82,"Bullish 9>21>50")
    elif e9<e21<e50: s("EMA Stack","down",82,"Bearish 9<21<50")
    elif e9>e21:     s("EMA Stack","up",  48,"Short bullish")
    else:            s("EMA Stack","down",48,"Short bearish")
    sk=ind["stoch_k"]
    if   sk<20: s("Stochastic","up",  80,f"Oversold ({sk:.0f})")
    elif sk>80: s("Stochastic","down",80,f"Overbought ({sk:.0f})")
    elif sk>ind["stoch_d"]: s("Stochastic","up",  42,"K above D")
    else:                   s("Stochastic","down",42,"K below D")
    cl=ind["close"]
    if   cl>ind["vwap"]*1.001: s("VWAP","up",  65,"Above VWAP")
    elif cl<ind["vwap"]*0.999: s("VWAP","down",65,"Below VWAP")
    else:                       s("VWAP","neut",0, "At VWAP")
    if ind["obv_slope"]>0: s("OBV","up",  60,"Rising OBV")
    else:                   s("OBV","down",60,"Falling OBV")
    vr=ind["vol_ratio"]
    if   vr>1.3: s("Volume","up",  72,f"Buy pressure {vr:.1f}x")
    elif vr<0.7: s("Volume","down",72,f"Sell pressure {1/vr:.1f}x")
    else:        s("Volume","neut",0, "Balanced")
    imb=ob.get("imbalance",0)
    if   imb>8:  s("Order Book","up",  74,f"Bid-heavy +{imb:.1f}%")
    elif imb<-8: s("Order Book","down",74,f"Ask-heavy {imb:.1f}%")
    else:        s("Order Book","neut",0, f"Balanced {imb:.1f}%")
    fg=sent.get("fg_value",50)
    if   fg<25:                  s("Sentiment","up",  66,"Extreme Fear")
    elif fg>75:                  s("Sentiment","down",66,"Extreme Greed")
    elif sent["signal"]=="up":   s("Sentiment","up",  44,"Bullish news")
    elif sent["signal"]=="down": s("Sentiment","down",44,"Bearish news")
    else:                        s("Sentiment","neut",0, "Neutral")
    up_sc  =sum(x["strength"] for x in sigs if x["dir"]=="up")
    dn_sc  =sum(x["strength"] for x in sigs if x["dir"]=="down")
    total  =up_sc+dn_sc+1e-9
    if   up_sc>dn_sc: return {"direction":"UP",  "cls":"up",  "confidence":round(up_sc/total*100),"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}
    elif dn_sc>up_sc: return {"direction":"DOWN","cls":"down","confidence":round(dn_sc/total*100),"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}
    else:             return {"direction":"WAIT","cls":"wait","confidence":50,"up_score":round(up_sc),"down_score":round(dn_sc),"signals":sigs}

def build_chart(ind):
    df=ind.get("df")
    if df is None or df.empty: return go.Figure()
    ts=df["ts"]
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.50,0.18,0.16,0.16],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=ts,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="BTC",increasing_fillcolor="#00e676",increasing_line_color="#00e676",
        decreasing_fillcolor="#ff5252",decreasing_line_color="#ff5252",line=dict(width=1)),row=1,col=1)
    for vals,color,name in [(ind["ema9_s"],"#ff8c00","EMA9"),(ind["ema21_s"],"#00c9ff","EMA21"),(ind["ema50_s"],"#b967ff","EMA50")]:
        fig.add_trace(go.Scatter(x=ts,y=vals,name=name,line=dict(color=color,width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["bb_up_s"],name="BB+",showlegend=False,
        line=dict(color="rgba(100,150,255,0.4)",width=1,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["bb_dn_s"],name="BB-",fill="tonexty",
        fillcolor="rgba(100,150,255,0.05)",showlegend=False,
        line=dict(color="rgba(100,150,255,0.4)",width=1,dash="dot")),row=1,col=1)
    up_mask=[df["close"].iloc[i]>=df["open"].iloc[i] for i in range(len(df))]
    vol_colors=["rgba(0,230,118,0.5)" if u else "rgba(255,82,82,0.5)" for u in up_mask]
    fig.add_trace(go.Bar(x=ts,y=df["volume"],name="Vol",marker_color=vol_colors,showlegend=False),row=2,col=1)
    hist_colors=["rgba(0,230,118,0.7)" if v>=0 else "rgba(255,82,82,0.7)" for v in ind["hist_s"]]
    fig.add_trace(go.Bar(x=ts,y=ind["hist_s"],name="Hist",marker_color=hist_colors,showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["macd_s"],name="MACD",line=dict(color="#ff8c00",width=1.2)),row=3,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["signal_s"],name="Signal",line=dict(color="#00c9ff",width=1.2)),row=3,col=1)
    fig.add_trace(go.Scatter(x=ts,y=ind["rsi_s"],name="RSI",line=dict(color="#b967ff",width=1.5)),row=4,col=1)
    for level,color in [(70,"rgba(255,82,82,0.35)"),(30,"rgba(0,230,118,0.35)"),(50,"rgba(255,255,255,0.1)")]:
        fig.add_hline(y=level,line_dash="dot",line_color=color,row=4,col=1)
    fig.update_layout(height=520,plot_bgcolor="#080c14",paper_bgcolor="#05080f",
        font=dict(family="Space Mono,monospace",size=9,color="#5a7a8c"),
        margin=dict(l=0,r=0,t=10,b=0),xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,font_size=9,bgcolor="rgba(0,0,0,0)"))
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#0f1825",zeroline=False,showticklabels=(i==4),row=i,col=1)
        fig.update_yaxes(gridcolor="#0f1825",zeroline=False,row=i,col=1)
    return fig

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ₿ BTC 5-Min Bot")
    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    candle_limit = st.slider("Candles", 100, 500, 200, 50)
    st.markdown("---")
    st.markdown("**Signals**")
    use_tech = st.checkbox("Technical (RSI/MACD/BB/EMA)", value=True)
    use_ob   = st.checkbox("Order Book", value=True)
    use_sent = st.checkbox("Sentiment",  value=True)
    st.markdown("---")
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:#2a3a4c;line-height:2;">⚠ NOT FINANCIAL ADVICE<br>Educational use only.</div>', unsafe_allow_html=True)

# ── HEADER ──
now_utc=datetime.now(timezone.utc)
secs_into=( now_utc.minute%5)*60+now_utc.second
secs_left=300-secs_into; m_l=secs_left//60; s_l=secs_left%60
st.markdown(f"""
<div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:#f0f8ff;'>
  ₿TC <span style='color:#ff8c00;'>5-Minute</span> Prediction Bot
</div>
<div style='font-family:Space Mono,monospace;font-size:10px;color:#2a3a4c;text-transform:uppercase;letter-spacing:2px;margin:4px 0 20px;'>
  Polymarket Signal Scanner &nbsp;·&nbsp; Live Kraken Data &nbsp;·&nbsp; 10 Indicators
</div>""", unsafe_allow_html=True)
col_btn,col_time,_=st.columns([1,2,5])
with col_btn: refresh=st.button("⟳ Refresh Now",use_container_width=True)
with col_time:
    st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:10px;color:#2a3a4c;padding-top:12px;">{now_utc.strftime("%H:%M:%S UTC")} &nbsp;·&nbsp; <span style="color:#00c9ff;">Next candle in {m_l}m {s_l:02d}s</span></div>',unsafe_allow_html=True)
st.markdown("---")

# ── FETCH ──
with st.spinner("Fetching live data..."):
    df_5m=fetch_candles(candle_limit); ticker=fetch_ticker()
    ob=fetch_orderbook(); fg=fetch_fear_greed()
    news=fetch_news() if use_sent else []; pm=fetch_polymarket_btc()

ind=compute_indicators(df_5m)
oba=analyze_orderbook(ob)
sent=analyze_sentiment(news,fg)
pred=predict(ind if use_tech else {}, oba if use_ob else {}, sent if use_sent else {})
price=ticker.get("price",0); pct_chg=ticker.get("pct_chg",0)
high_24=ticker.get("high",0); low_24=ticker.get("low",0); vol_24=ticker.get("vol_24h",0)

# ── METRICS ──
m1,m2,m3,m4,m5=st.columns(5)
with m1: st.metric("BTC/USD",f"${price:,.2f}",f"{pct_chg:+.2f}%")
with m2: st.metric("24h High",f"${high_24:,.0f}")
with m3: st.metric("24h Low",f"${low_24:,.0f}")
with m4: st.metric("24h Volume",f"{vol_24:,.1f} BTC")
with m5: st.metric("Fear & Greed",f"{fg['value']} — {fg['label']}",f"{fg['value']-fg['prev']:+d}")
st.markdown("---")

# ── CHART + SIGNAL ──
chart_col,signal_col=st.columns([3,1])
with chart_col:
    st.markdown("#### 📈 5m BTC/USD")
    if ind: st.plotly_chart(build_chart(ind),use_container_width=True,config={"displayModeBar":False})
    else: st.warning("Need 60+ candles to render chart.")
with signal_col:
    st.markdown("#### ⚡ Signal")
    cls=pred["cls"]; arrow="▲" if cls=="up" else ("▼" if cls=="down" else "◆")
    st.markdown(f"""
    <div class="big-signal {cls}">
      <div class="signal-arrow {cls}">{arrow}</div>
      <div class="signal-word">{pred['direction']}</div>
      <div class="signal-conf {cls}">{pred['confidence']}%</div>
      <div class="signal-sub">confidence</div>
    </div>""", unsafe_allow_html=True)
    up_w=min(pred.get("up_score",0)/7,100); dn_w=min(pred.get("down_score",0)/7,100)
    st.markdown(f"""
    <div style='margin-bottom:14px;'>
      <div class='ind-row'><span class='ind-name'>UP</span>
        <div style='flex:1;margin:0 8px;height:5px;background:#0f1825;border-radius:3px;overflow:hidden;'>
          <div style='width:{up_w}%;height:100%;background:#00e676;border-radius:3px;'></div></div>
        <span class='ind-val up'>{pred.get("up_score",0)}</span></div>
      <div class='ind-row'><span class='ind-name'>DOWN</span>
        <div style='flex:1;margin:0 8px;height:5px;background:#0f1825;border-radius:3px;overflow:hidden;'>
          <div style='width:{dn_w}%;height:100%;background:#ff5252;border-radius:3px;'></div></div>
        <span class='ind-val down'>{pred.get("down_score",0)}</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("**Signal Breakdown**")
    rows=""
    for sig in pred["signals"]:
        d=sig["dir"]; icon="▲" if d=="up" else("▼" if d=="down" else "◆")
        c2="up" if d=="up" else("down" if d=="down" else "neut")
        rows+=f"<div class='ind-row'><span class='ind-name'>{sig['name']}</span><span class='ind-val {c2}'>{icon} {sig['reason']}</span></div>"
    st.markdown(f"<div style='background:#080c14;border:1px solid #0f1825;border-radius:10px;'>{rows}</div>",unsafe_allow_html=True)

# ── INDICATOR VALUES ──
if ind:
    st.markdown("---")
    st.markdown("#### 📐 Indicator Values")
    i1,i2,i3,i4,i5,i6=st.columns(6)
    with i1: st.metric("RSI-14",     f"{ind['rsi']:.1f}")
    with i2: st.metric("MACD Hist",  f"{ind['macd_hist']:.2f}")
    with i3: st.metric("BB Position",f"{ind['bb_pos']*100:.0f}%")
    with i4: st.metric("Stoch %K",   f"{ind['stoch_k']:.1f}")
    with i5: st.metric("VWAP",       f"${ind['vwap']:,.0f}")
    with i6: st.metric("ATR-14",     f"${ind['atr']:,.0f}")
    i7,i8,i9,i10,i11,i12=st.columns(6)
    with i7:  st.metric("EMA-9",    f"${ind['ema9']:,.0f}")
    with i8:  st.metric("EMA-21",   f"${ind['ema21']:,.0f}")
    with i9:  st.metric("EMA-50",   f"${ind['ema50']:,.0f}")
    with i10: st.metric("Vol Ratio",f"{ind['vol_ratio']:.2f}x")
    with i11: st.metric("OBV Slope",f"{ind['obv_slope']:+,.0f}")
    with i12: st.metric("OB Imbal", f"{oba['imbalance']:+.1f}%")

# ── ORDER BOOK + SENTIMENT ──
st.markdown("---")
ob_col,sent_col=st.columns(2)
with ob_col:
    st.markdown("#### 📖 Order Book")
    bids=ob.get("bids",[])[:25]; asks=ob.get("asks",[])[:25]
    if bids and asks:
        bid_cum=np.cumsum([q for _,q in bids])[::-1]; ask_cum=np.cumsum([q for _,q in asks])
        fig_ob=go.Figure()
        fig_ob.add_trace(go.Scatter(x=[p for p,_ in bids][::-1],y=bid_cum,fill="tozeroy",name="Bids",
            line=dict(color="#00e676",width=1.5),fillcolor="rgba(0,230,118,0.12)"))
        fig_ob.add_trace(go.Scatter(x=[p for p,_ in asks],y=ask_cum,fill="tozeroy",name="Asks",
            line=dict(color="#ff5252",width=1.5),fillcolor="rgba(255,82,82,0.12)"))
        fig_ob.update_layout(height=250,plot_bgcolor="#080c14",paper_bgcolor="#05080f",
            margin=dict(l=0,r=0,t=10,b=0),font=dict(family="Space Mono,monospace",size=9,color="#5a7a8c"),
            legend=dict(orientation="h",font_size=9,bgcolor="rgba(0,0,0,0)"))
        fig_ob.update_xaxes(gridcolor="#0f1825",zeroline=False)
        fig_ob.update_yaxes(gridcolor="#0f1825",zeroline=False)
        st.plotly_chart(fig_ob,use_container_width=True,config={"displayModeBar":False})
with sent_col:
    st.markdown("#### 🧠 Sentiment")
    fg_val=fg["value"]
    fg_color="#ff5252" if fg_val<25 else "#ff8c00" if fg_val<45 else "#ffeb3b" if fg_val<55 else "#69f0ae" if fg_val<75 else "#00e676"
    fig_fg=go.Figure(go.Indicator(mode="gauge+number",value=fg_val,
        number={"font":{"color":fg_color,"family":"Space Mono","size":30}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#2a3a4c","tickfont":{"size":9}},
               "bar":{"color":fg_color,"thickness":0.25},"bgcolor":"#080c14","bordercolor":"#0f1825",
               "steps":[{"range":[0,25],"color":"rgba(255,82,82,0.12)"},{"range":[25,45],"color":"rgba(255,140,0,0.08)"},
                        {"range":[45,55],"color":"rgba(255,235,59,0.06)"},{"range":[55,75],"color":"rgba(105,240,174,0.08)"},
                        {"range":[75,100],"color":"rgba(0,230,118,0.12)"}]},
        title={"text":f"Fear & Greed · {fg['label']}","font":{"color":"#5a7a8c","size":10,"family":"Space Mono"}}))
    fig_fg.update_layout(height=200,paper_bgcolor="#05080f",margin=dict(l=20,r=20,t=30,b=0))
    st.plotly_chart(fig_fg,use_container_width=True,config={"displayModeBar":False})
    if news:
        st.markdown("**Latest BTC News**")
        for n in news[:4]:
            icon="▲" if n["sent"]=="up" else ("▼" if n["sent"]=="down" else "◆")
            color="#00e676" if n["sent"]=="up" else ("#ff5252" if n["sent"]=="down" else "#ff8c00")
            st.markdown(f'<div class="pm-box" style="padding:10px 12px;margin-bottom:6px;"><div style="font-size:12px;font-weight:600;color:#c8d8e8;line-height:1.4;margin-bottom:4px;">{n["title"][:85]}{"..." if len(n["title"])>85 else ""}</div><div style="font-family:Space Mono,monospace;font-size:9px;color:{color};">{icon} {n["sent"].upper()} &nbsp;·&nbsp; 👍{n["bull"]} 👎{n["bear"]}</div></div>',unsafe_allow_html=True)

# ── POLYMARKET MARKETS ──
st.markdown("---")
st.markdown("#### 🎯 Active Polymarket BTC Markets")
pred_cls=pred["cls"]
if pm:
    for m in pm:
        question=m.get("question") or m.get("title") or "Unknown"
        cid=m.get("conditionId") or m.get("id") or ""
        tokens=m.get("tokens",[]) or []
        yes_p=no_p=None
        for t in tokens:
            if not isinstance(t,dict): continue
            out=(t.get("outcome") or "").upper(); p=t.get("price") or t.get("lastTradePrice")
            if p:
                if "YES" in out: yes_p=float(p)
                if "NO"  in out: no_p=float(p)
        q_low=question.lower()
        is_up=any(w in q_low for w in ["above","higher","exceed","rise","bull","up"])
        is_dn=any(w in q_low for w in ["below","lower","fall","drop","crash","bear","down"])
        if   (pred_cls=="up" and is_up) or (pred_cls=="down" and is_dn): align="up";   atxt="✦ ALIGNED"
        elif (pred_cls=="up" and is_dn) or (pred_cls=="down" and is_up): align="down"; atxt="✗ AGAINST"
        else:                                                              align="neut"; atxt="◆ NEUTRAL"
        hrs=""
        try:
            end=m.get("endDate") or m.get("end_date") or ""
            if end:
                edt=datetime.fromisoformat(str(end).replace("Z","+00:00"))
                hl=(edt-datetime.now(timezone.utc)).total_seconds()/3600
                hrs=f"{hl:.1f}h"
        except: pass
        yes_profit=f"+{round(((1-yes_p)/yes_p)*100,1)}%" if yes_p and 0.01<yes_p<0.95 else "—"
        st.markdown(f"""
        <div class='pm-box'>
          <div class='pm-q'>{question}</div>
          <div class='pm-row'>
            <div class='pm-metric'><div class='pm-label'>YES</div><div class='pm-value'>{f'{yes_p:.3f}' if yes_p else '—'}</div></div>
            <div class='pm-metric'><div class='pm-label'>NO</div><div class='pm-value'>{f'{no_p:.3f}' if no_p else '—'}</div></div>
            <div class='pm-metric'><div class='pm-label'>Closes</div><div class='pm-value'>{hrs or '—'}</div></div>
            <div class='pm-metric'><div class='pm-label'>YES Profit</div><div class='pm-value up'>{yes_profit}</div></div>
            <div class='pm-metric'><div class='pm-label'>Signal</div><div class='pm-value {align}'>{atxt}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
        if cid: st.markdown(f'<a href="https://polymarket.com/event/{cid}" target="_blank" style="font-family:Space Mono,monospace;font-size:10px;color:#00c9ff;text-decoration:none;">↗ Open on Polymarket</a>',unsafe_allow_html=True)
else:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:12px;color:#2a3a4c;padding:20px;">No active BTC markets found on Polymarket.</div>',unsafe_allow_html=True)

if auto_refresh:
    time.sleep(30)
    st.rerun()
