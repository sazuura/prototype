import pandas as pd

pairs = ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]
dfs   = []

for p in pairs:
    try:
        df = pd.read_csv(f"data/{p}/labeled.csv")
        df["pair"] = p
        dfs.append(df)
    except FileNotFoundError:
        pass

all_df         = pd.concat(dfs, ignore_index=True)
all_df["time"] = pd.to_datetime(all_df["time"])

# 1. Filter hanya kandidat momentum
df_ml = all_df[all_df["momentum_candidate"] == True].copy()

# 2. Drop kolom yang tidak dipakai ML / data leakage
drop_cols = [
    # Raw OHLC — leakage
    "open", "high", "low", "close",
    # EMA absolut — leakage (pakai dist_to_ema & slope sebagai gantinya)
    "ema20", "ema50", "ema200",
    # Harga absolut swing — leakage antar pair  [FIX #2]
    "swing_high_20", "swing_low_20",
    # Kolom helper yang sudah terwakili fitur lain
    "tr", "body", "upper_wick", "lower_wick", "candle_range",
    # Kolom pipeline — bukan fitur
    "close_future", "momentum_candidate", "cand_long", "cand_short",
]
df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

# 3. Pastikan semua numerik
non_numeric = [c for c in df_ml.select_dtypes(exclude=["number"]).columns
               if c not in ["time", "pair"]]
if non_numeric:
    df_ml = df_ml.drop(columns=non_numeric)

df_ml.to_csv("data/dataset_ml.csv", index=False)

feature_cols = [c for c in df_ml.columns if c not in ["label","time","pair"]]
long_count   = (df_ml.get("direction", pd.Series()) ==  1).sum()
short_count  = (df_ml.get("direction", pd.Series()) == -1).sum()

print(f"Dataset ML ready!")
print(f"  Total kandidat : {len(df_ml):,}  (LONG: {long_count} | SHORT: {short_count})")
print(f"  WR overall     : {df_ml['label'].mean():.2%}")
print(f"  Jumlah fitur   : {len(feature_cols)}")
print(f"  Fitur          : {feature_cols}")