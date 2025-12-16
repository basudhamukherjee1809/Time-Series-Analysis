import yfinance as yf
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31", interval="1d")
data.to_csv(data.to_csv(r"C:\Users\basud\Downloads\aapl_daily.csv"))
