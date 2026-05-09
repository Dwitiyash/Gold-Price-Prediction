import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Gold Price Prediction Dashboard", page_icon="🥇", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.main {
    background-color:#0b1120;
}
div[data-testid="metric-container"]{
    background:#071633;
    border:1px solid #1f3b73;
    padding:18px;
    border-radius:12px;
}
h1,h2,h3 {
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data(ttl=1800)
def load_data():
    gold = yf.download("GC=F", period="5y", progress=False)
    sp500 = yf.download("^GSPC", period="5y", progress=False)
    usd = yf.download("DX-Y.NYB", period="5y", progress=False)
    tnx = yf.download("^TNX", period="5y", progress=False)

    gold = gold.add_suffix("_gold")
    sp500 = sp500.add_suffix("_sp500")
    usd = usd.add_suffix("_usd")
    tnx = tnx.add_suffix("_tnx")

    data = gold.join(sp500).join(usd).join(tnx)
    data.columns = data.columns.get_level_values(0)

    return data

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    with open("model_dir.pkl", "rb") as f:
        model_dir = pickle.load(f)

    with open("model_vol.pkl", "rb") as f:
        model_vol = pickle.load(f)

    return model_dir, model_vol

# ---------- MAIN ----------
st.title("🥇 Gold Price Prediction Dashboard")

data = load_data()
model_dir, model_vol = load_models()

# ---------- FEATURE ENGINEERING ----------
data['gold_diff'] = data['Close_gold'] - data['Close_gold'].shift(1)
data['yield_change'] = data['Close_tnx'].pct_change()

data['gold_momentum3'] = data['Close_gold'] - data['Close_gold'].shift(3)

data['gold_range'] = data['High_gold'] - data['Low_gold']
data['gold_range_pct'] = data['gold_range'] / data['Close_gold']

data['ma7'] = data['Close_gold'].rolling(7).mean()
data['ma21'] = data['Close_gold'].rolling(21).mean()
data['trend'] = data['ma7'] - data['ma21']

delta = data['Close_gold'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['rsi'] = 100 - (100 / (1 + rs))

data['gold_usd_ratio'] = data['Close_gold'] / data['Close_usd']
data['gold_sp500_ratio'] = data['Close_gold'] / data['Close_sp500']

data['volatility_7'] = data['Close_gold'].rolling(7).std()
data['volatility_14'] = data['Close_gold'].rolling(14).std()

data = data.dropna()

# ---------- FEATURES ----------
features_dir = [
    'gold_diff',
    'gold_momentum3',
    'trend',
    'rsi',
    'gold_usd_ratio',
    'gold_sp500_ratio'
]

features_vol = [
    'gold_range_pct',
    'gold_diff',
    'volatility_7',
    'volatility_14',
    'gold_range',
    'yield_change'
]

X_dir = data[features_dir].iloc[-1:]
X_vol = data[features_vol].iloc[-1:]

# ---------- PREDICTION ----------
direction = model_dir.predict(X_dir)[0]
volatility = model_vol.predict(X_vol)[0]

current_price = data['Close_gold'].iloc[-1]

trend = "UP" if direction == 1 else "DOWN"

move_amount = current_price * volatility

predicted_price = current_price + move_amount if direction == 1 else current_price - move_amount

# ---------- DATE INFO ----------
latest_date = data.index[-1].date()
next_date = latest_date + pd.Timedelta(days=1)

# ---------- TOP METRICS ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", round(current_price,2))
    st.caption(f"Latest Close - {latest_date}")

with col2:
    st.metric("Prediction", trend)
    st.caption(f"For {next_date}")

with col3:
    st.metric("Expected Move %", str(round(volatility*100,2)) + "%")
    st.caption("Estimated Move")

with col4:
    st.metric("Predicted Price", round(predicted_price,2))
    st.caption(f"Forecast Close - {next_date}")

# ---------- TABLE + CHART ----------
col5, col6 = st.columns(2)

with col5:
    st.subheader("Historical Gold Prices")
    show_df = data[['Close_gold','ma7','ma21','rsi']].tail(15).copy()
    show_df.columns = ['Close','MA7','MA21','RSI']
    show_df.index = show_df.index.date
    show_df.index.name = "Date"

    st.dataframe(show_df[::-1], use_container_width=True)

with col6:
    st.subheader("Gold Price Trend")
    st.line_chart(data['Close_gold'].tail(365))

# ---------- SNAPSHOT ----------
st.markdown("---")
st.subheader("Recent Market Snapshot")

c1, c2, c3, c4 = st.columns(4)

day_change = ((data['Close_gold'].iloc[-1] - data['Close_gold'].iloc[-2]) / data['Close_gold'].iloc[-2]) * 100
five_day = ((data['Close_gold'].iloc[-1] - data['Close_gold'].iloc[-6]) / data['Close_gold'].iloc[-6]) * 100

c1.metric("1 Day Change", f"{day_change:.2f}%")
c2.metric("5 Day Return", f"{five_day:.2f}%")
c3.metric("7 Day Avg", f"{data['ma7'].iloc[-1]:,.2f}")
c4.metric("21 Day Avg", f"{data['ma21'].iloc[-1]:,.2f}")

c5, c6, c7 = st.columns(3)

high30 = data['High_gold'].tail(30).max()
low30 = data['Low_gold'].tail(30).min()
rsi_now = data['rsi'].iloc[-1]

if rsi_now > 70:
    rsi_status = "Overbought"
elif rsi_now < 30:
    rsi_status = "Oversold"
else:
    rsi_status = "Neutral"

c5.metric("30 Day High", f"{high30:,.2f}")
c6.metric("30 Day Low", f"{low30:,.2f}")
c7.metric("RSI Status", rsi_status)

st.markdown("---")
st.caption("Prediction is based on latest available close data and estimates next trading day close.")