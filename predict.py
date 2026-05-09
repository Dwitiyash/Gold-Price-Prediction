import yfinance as yf
import pandas as pd
import numpy as np
import pickle

gold = yf.download("GC=F", period="5y")
sp500 = yf.download("^GSPC", period="5y")
usd = yf.download("DX-Y.NYB", period="5y")
tnx = yf.download("^TNX", period="5y")

gold = gold.add_suffix('_gold')
sp500 = sp500.add_suffix('_sp500')
usd = usd.add_suffix('_usd')
tnx = tnx.add_suffix('_tnx')

data = gold.join(sp500).join(usd).join(tnx)
data.columns = data.columns.get_level_values(0)

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

with open("model_dir.pkl", "rb") as f:
    model_dir = pickle.load(f)

with open("model_vol.pkl", "rb") as f:
    model_vol = pickle.load(f)

direction = model_dir.predict(X_dir)[0]
volatility = model_vol.predict(X_vol)[0]

current_price = data['Close_gold'].iloc[-1]

trend = "UP" if direction == 1 else "DOWN"

move_amount = current_price * volatility

predicted_price = current_price + move_amount if direction == 1 else current_price - move_amount

print("Current Price:", round(current_price,2))
print("Prediction:", trend)
print("Expected Move %:", round(volatility*100,2),"%")
print("Predicted Price:", round(predicted_price,2))