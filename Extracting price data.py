import os
print(os.getcwd())
import yfinance as yf

tickers = {
    "JSW_Infrastructure": "JSWINFRA.NS",
    "Ester_Industries": "ESTER.NS",
    "Polycab_India": "POLYCAB.NS",
    "Trent": "TRENT.NS",
    "Sasken_Tech": "SASKEN.NS",
    "Bharti_Airtel": "BHARTIARTL.NS",
    "Cera_Sanitary": "CERA.NS",
    "Supreme_Inds": "SUPREMEIND.NS",
    "Alkyl_Amines": "ALKYLAMINE.NS",
    "Deepak_Nitrite": "DEEPAKNTR.NS",
    "Divis_Lab": "DIVISLAB.NS",
    "Balaji_Amines": "BALAMINES.NS",
    "Indostar_Capital": "INDOSTAR.NS",
    "SKF_India": "SKFINDIA.NS"
}

start_date = "2020-01-01"
end_date   = "2024-12-31"

for name, ticker in tickers.items():
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False
    )
    
    if not data.empty:
        data.to_csv(f"{name.lower()}_daily.csv")
        print(f"{name}: saved successfully")
    else:
        print(f"{name}: no data found")

