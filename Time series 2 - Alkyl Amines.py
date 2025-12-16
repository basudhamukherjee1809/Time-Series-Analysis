import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                       module="statsmodels.base.model")

# ---- 1. Load CSV properly ----
# Skip the first 3 metadata rows
df = pd.read_csv(
    r"C:\Users\basud\AppData\Local\Programs\Microsoft VS Code\alkyl_amines_daily.csv",
    skiprows=3,
    names=["Date", "Close", "High", "Low", "Open", "Volume"],
    parse_dates=["Date"]
)

# Set Date as index for time series plotting
df.set_index("Date", inplace=True)
#Moves the Date column into the DataFrame index, each row is now identified by a date, because Time-series in Pandas are index-based, not column-based.

# ---- 2. Plot multiple time series on the same axes ----
plt.figure(figsize=(12, 6))

plt.plot(df.index, df["Close"], label="Close")
plt.plot(df.index, df["High"], label="High")
plt.plot(df.index, df["Low"], label="Low")
plt.plot(df.index, df["Open"], label="Open")

# ---- 3. Plot formatting ----
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Alkyl Amines Daily Price Time Series")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Log-returns (standard in finance)
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

df["rolling_mean"] = df["Close"].rolling(50).mean()
df["rolling_vol"] = df["log_return"].rolling(50).std()



result = adfuller(df["log_return"])
print(f"ADF statistic: {result[0]:.20f}")
print(f"p-value: {result[1]:.20f}")

# Add frequency and scale data
df = df.asfreq('B')  # Business days
model = ARIMA(df['log_return'], order=(1,0,1))  # No constant
fit = model.fit()  # fit is a results object, containing: Estimated coefficients, standard errors, residuals, information criteria (AIC, BIC)
print(fit.summary())
forecast = fit.forecast(steps=20)  #Computes expected future log-returns for the next 20 trading days
print(forecast)

last_price = df["Close"].iloc[-1]
price_forecast = last_price * np.exp(np.cumsum(forecast))

future_dates = pd.date_range(
    start=df.index[-1],
    periods=len(price_forecast) + 1,
    freq="B"
)[1:]

plt.plot(df.index, df["Close"], label="Historical")
plt.plot(future_dates, price_forecast, label="Forecast", linestyle="--")
plt.legend()
plt.show()


# ---------------- GARCH VOLATILITY FORECAST ---------------- #

# Ensure strictly finite values
df = df[np.isfinite(df["log_return"])]

# Fit GARCH(1,1)
garch = arch_model(
    df["log_return"] * 100,  # scale for numerical stability
    vol="Garch",
    p=1,
    q=1
)

garch_fit = garch.fit(disp="off")

# Forecast variance for next 20 days
vol_forecast = garch_fit.forecast(horizon=20)

# Extract variance forecast for the last observed time
variance_forecast = vol_forecast.variance.iloc[-1]

# Convert variance → volatility
volatility_forecast = np.sqrt(variance_forecast)

# Create future dates
future_dates = pd.date_range(
    start=df.index[-1],
    periods=len(volatility_forecast) + 1,
    freq="B"
)[1:]

# Plot historical and forecasted volatility
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["rolling_vol"] * 100,
    label="Historical Volatility (Rolling)",
)
plt.plot(
    future_dates,
    volatility_forecast,
    linestyle="--",
    label="Forecast Volatility (GARCH)"
)

plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.title("Alkyl Amines Volatility Forecast using GARCH(1,1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------- MONTE CARLO PRICE PATH SIMULATION ---------------- #

n_steps = 20      # same horizon as before
n_paths = 300     # number of simulated price paths

last_price = df["Close"].iloc[-1]

# ARIMA mean forecast (log-returns)
mean_forecast = forecast.values  # already computed earlier

# GARCH volatility forecast (convert back to return scale)
sigma = volatility_forecast.values / 100.0

# Container for price paths
price_paths = np.zeros((n_steps + 1, n_paths))
price_paths[0, :] = last_price

# Monte Carlo simulation
for i in range(n_paths):
    shocks = np.random.normal(0, 1, n_steps)
    simulated_returns = mean_forecast + sigma * shocks
    price_paths[1:, i] = last_price * np.exp(np.cumsum(simulated_returns))

# ---------------- PLOT ---------------- #

plt.figure(figsize=(12, 6))

# Monte Carlo paths
for i in range(price_paths.shape[1]):
    plt.plot(future_dates, price_paths[1:, i], color="gray", alpha=0.05)

# Historical prices
plt.plot(df.index, df["Close"], color="blue", label="Historical")

# Deterministic ARIMA mean forecast
plt.plot(future_dates, price_forecast, "r--", linewidth=2, label="Mean Forecast")

plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Alkyl Amines Monte Carlo Price Paths (ARIMA + GARCH)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
