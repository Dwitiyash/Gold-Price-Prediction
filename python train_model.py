import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
gold = yf.download("GC=F", period="5y")
sp500 = yf.download("^GSPC", period="5y")
usd = yf.download("DX-Y.NYB", period="5y")
tnx = yf.download("^TNX", period="5y")

gold = gold.add_suffix('_gold')
sp500 = sp500.add_suffix('_sp500')
usd = usd.add_suffix('_usd')
tnx=tnx.add_suffix('_tnx')

data = gold.join(sp500).join(usd).join(tnx)

data.columns = data.columns.get_level_values(0)

# Add to your features list
data['Gold_return'] = data['Close_gold'].pct_change()
data['sp500_return'] = data['Close_sp500'].pct_change()
data['usd_return'] = data['Close_usd'].pct_change()
data['gold_diff'] = data['Close_gold'] - data['Close_gold'].shift(1)
data['yield_change'] = data['Close_tnx'].pct_change()

# Momentum
data['gold_momentum3'] = data['Close_gold'] - data['Close_gold'].shift(3)

# Volatility
data['gold_range'] = data['High_gold'] - data['Low_gold']
data['gold_range_pct'] = data['gold_range'] / data['Close_gold']
data['ma7']=data['Close_gold'].rolling(window=7).mean()
data['ma21']=data['Close_gold'].rolling(window=21).mean()
data['trend']=data['ma7'] - data['ma21']
delta=data['Close_gold'].diff()
gain=(delta.where(delta>0,0)).rolling(window=14).mean()
loss=(-delta.where(delta<0,0)).rolling(window=14).mean()
rs=gain/loss
data['rsi']=100-(100/(1+rs))
# Ratios (macro signals)
data['gold_usd_ratio'] = data['Close_gold'] / data['Close_usd']
data['gold_sp500_ratio'] = data['Close_gold'] / data['Close_sp500']
data['gold_tnx_ratio'] = data['Close_gold'] / data['Close_tnx']
future_price = data['Close_gold'].shift(-3)
current_price = data['Close_gold']

future_price = data['Close_gold'].shift(-3)
current_price = data['Close_gold']
data['volatility_7'] = data['Close_gold'].rolling(7).std()
data['volatility_14'] = data['Close_gold'].rolling(14).std()
data['target_direction'] = (
    data['Close_gold'].shift(-1) > data['Close_gold']
).astype(int)
data['target_volatility'] = abs(
    (data['Close_gold'].shift(-1) - data['Close_gold']) / data['Close_gold']
)
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

X_dir = data[features_dir]
X_vol = data[features_vol]
y_dir = data['target_direction']
y_vol = data['target_volatility']
train_size = int(len(data) * 0.8)

train_size = int(len(data) * 0.8)

X_dir_train = X_dir[:train_size]
X_dir_test = X_dir[train_size:]

X_vol_train = X_vol[:train_size]
X_vol_test = X_vol[train_size:]

y_dir_train = y_dir[:train_size]
y_dir_test = y_dir[train_size:]

y_vol_train = y_vol[:train_size]
y_vol_test = y_vol[train_size:]

model_dir = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_vol = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_dir.fit(X_dir_train, y_dir_train)
model_vol.fit(X_vol_train, y_vol_train)
pred_dir = model_dir.predict(X_dir_test)
pred_vol = model_vol.predict(X_vol_test)
print("Direction Accuracy:", round(accuracy_score(y_dir_test, pred_dir),4))
print("Volatility MAE:", round(mean_absolute_error(y_vol_test, pred_vol),4))
model_dir.fit(X_dir, y_dir)
model_vol.fit(X_vol, y_vol)
import pickle
# Save direction model
with open("model_dir.pkl", "wb") as f:
    pickle.dump(model_dir, f)

# Save volatility model
with open("model_vol.pkl", "wb") as f:
    pickle.dump(model_vol, f)

print(pd.Series(model_dir.feature_importances_, index=features_dir).sort_values(ascending=False))
print(pd.Series(model_vol.feature_importances_, index=features_vol).sort_values(ascending=False))

print("Models saved successfully!")
# result_analysis.py
# Run this after training your Gold Price Prediction model

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# USE YOUR ACTUAL VALUES HERE
# ---------------------------------------------------

# Real gold prices (test data)
y_test = np.array([2310, 2325, 2338, 2342, 2350, 2361, 2375, 2382, 2390, 2401])

# Predicted prices by model
y_pred = np.array([2308, 2322, 2340, 2339, 2348, 2365, 2372, 2380, 2394, 2398])

# If using Random Forest / XGBoost:
feature_names = [
    "Open Price",
    "High Price",
    "Low Price",
    "Volume",
    "USD Index",
    "S&P 500",
    "Previous Close",
    "5 Day Avg",
    "10 Day Avg"
]

feature_importance = [0.21, 0.18, 0.15, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05]

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Gold Price Prediction Results")
print("-----------------------------")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,4))

# ---------------------------------------------------
# GRAPH 1: Actual vs Predicted
# ---------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Price", marker="o")
plt.plot(y_pred, label="Predicted Price", marker="o")
plt.title("Actual vs Predicted Gold Prices")
plt.xlabel("Days")
plt.ylabel("Gold Price")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# GRAPH 2: Feature Importance
# ---------------------------------------------------

plt.figure(figsize=(10,5))
sns.barplot(x=feature_importance, y=feature_names, palette="Blues_r")
plt.title("Most Important Features")
plt.xlabel("Importance Score")
plt.show()

# ---------------------------------------------------
# GRAPH 3: Error Distribution
# ---------------------------------------------------

errors = y_test - y_pred

plt.figure(figsize=(8,5))
sns.histplot(errors, bins=8, kde=True, color="green")
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.show()