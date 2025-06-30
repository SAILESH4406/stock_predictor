import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

# Step 1: Load some sample stock data using yfinance
import yfinance as yf
df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')  # Apple stock example

# Step 2: Select the column(s) to scale
data = df[['Close']]  # You can also scale multiple columns if needed

# Step 3: Fit the scaler
scaler = MinMaxScaler()
scaler.fit(data)

# Step 4: Save it using pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler created and saved as 'scaler.pkl'")
