import MetaTrader5 as mt5
import pandas as pd
import os
import time

# Config
pairs = ["EURUSDc","USDJPYc","GBPUSDc","USDCHFc","AUDUSDc","USDCADc","NZDUSDc"]
TIMEFRAME = mt5.TIMEFRAME_M15
TOTAL     = 160000

mt5.initialize()
os.makedirs("data", exist_ok=True)

for pair in pairs:
    clean = pair.replace("c", "")
    folder = f"data/{clean}"
    os.makedirs(folder, exist_ok=True)

    rates = mt5.copy_rates_from_pos(pair, TIMEFRAME, 1, TOTAL)
    if rates is None or len(rates) == 0:
        print(f"{pair} FAILED atau data tidak tersedia"); continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    before = len(df)
    
    # Cleaning & Filtering
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"]  <= df["open"]) &
            (df["low"]  <= df["close"]) &
            (df["tick_volume"] > 0)]
    df = df[df["time"].dt.dayofweek < 5].reset_index(drop=True)

    # Gap Detection (M15 = 900 detik)
    df["time_diff"] = df["time"].diff().dt.total_seconds() / 900  
    gap_mask = df["time_diff"] > 10
    if gap_mask.sum() > 0:
        print(f"  [{clean}] Gap terdeteksi: {gap_mask.sum()} titik")
    df = df.drop(columns=["time_diff"])

    after = len(df)
    df.to_csv(f"{folder}/raw.csv", index=False)
    print(f"{clean} | bars: {after} | "
          f"{df['time'].min()} → {df['time'].max()}")

mt5.shutdown()