import pandas as pd
import matplotlib.pyplot as plt

# ---- 1. Load CSV properly ----
# Skip the first 3 metadata rows
df = pd.read_csv(
    r"C:\Users\basud\Downloads\aapl_daily.csv",
    skiprows=3,
    names=["Date", "Close", "High", "Low", "Open", "Volume"],
    parse_dates=["Date"]
)

# Set Date as index for time series plotting
df.set_index("Date", inplace=True)

# ---- 2. Plot multiple time series on the same axes ----
plt.figure(figsize=(12, 6))

plt.plot(df.index, df["Close"], label="Close")
plt.plot(df.index, df["High"], label="High")
plt.plot(df.index, df["Low"], label="Low")
plt.plot(df.index, df["Open"], label="Open")

# ---- 3. Plot formatting ----
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("AAPL Daily Price Time Series")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()