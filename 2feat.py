import pandas as pd
import numpy as np
import time

pairs = ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]
BODY_ATR_MIN   = 0.8    
WICK_BODY_MAX  = 0.6    
BODY_RANGE_MIN = 0.45   
ATR_SL_MULT    = 2.0
RR_RATIO       = 1.0
MAX_HOLD       = 40

def build_features(df, pair):
    df["time"] = pd.to_datetime(df["time"])
    mask = (df["time"] >= "2020-01-01") & (df["time"] <= "2025-12-31 23:59:59")
    df   = df.loc[mask].copy()

    pip_mult = 0.001 if "JPY" in pair else 0.00001

    # Candle anatomy
    df["body"]         = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"]   = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"]   = df[["open","close"]].min(axis=1) - df["low"]
    df["candle_dir"]   = np.sign(df["close"] - df["open"])

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"]  - df["close"].shift()).abs()
    df["tr"]       = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    df["atr"]      = df["tr"].rolling(14).mean()
    df["atr_slow"] = df["tr"].rolling(50).mean()

    # Normalized — ATR based
    df["body_atr"]         = df["body"]       / df["atr"]
    df["upper_wick_atr"]   = df["upper_wick"] / df["atr"]
    df["lower_wick_atr"]   = df["lower_wick"] / df["atr"]
    df["spread_atr"]       = (df["spread"] * pip_mult) / df["atr"]
    df["body_range_ratio"] = df["body"] / df["candle_range"].replace(0, np.nan)

    # Wick vs body ratio (definisi momentum)
    df["upper_wick_body"] = df["upper_wick"] / df["body"].replace(0, np.nan)
    df["lower_wick_body"] = df["lower_wick"] / df["body"].replace(0, np.nan)

    # Volume
    df["vol_mean"]  = df["tick_volume"].rolling(50).mean()
    df["vol_ratio"] = df["tick_volume"] / df["vol_mean"]

    # EMA & slope
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["ema20_slope"]  = (df["ema20"]  - df["ema20"].shift(5))  / df["atr"]
    df["ema50_slope"]  = (df["ema50"]  - df["ema50"].shift(10)) / df["atr"]
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(20)) / df["atr"]

    df["micro_trend_atr"]   = (df["ema20"] - df["ema50"])  / df["atr"]
    df["macro_trend_atr"]   = (df["ema50"] - df["ema200"]) / df["atr"]
    df["dist_to_ema20_atr"] = (df["close"] - df["ema20"])  / df["atr"]
    df["dist_to_ema50_atr"] = (df["close"] - df["ema50"])  / df["atr"]

    # Posisi relatif terhadap EMA200 (fitur, bukan filter)
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)

    # RSI
    delta = df["close"].diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df["rsi"]        = 100 - (100 / (1 + rs))
    df["rsi_slope"]  = df["rsi"] - df["rsi"].shift(5)
    df["rsi_dist50"] = df["rsi"] - 50

    # MACD
    macd = (df["close"].ewm(span=12, adjust=False).mean()
          - df["close"].ewm(span=26, adjust=False).mean())
    df["macd_hist_atr"]   = (macd - macd.ewm(span=9, adjust=False).mean()) / df["atr"]
    df["macd_hist_slope"] = df["macd_hist_atr"] - df["macd_hist_atr"].shift(3)

    # Volatility regime
    df["vol_regime"] = df["atr"] / df["atr_slow"]
    df["atr_std"]    = df["tr"].rolling(50).std()
    df["vol_z_atr"]  = (df["atr"] - df["atr_slow"]) / df["atr_std"]

    # Range & position
    roll_high = df["high"].rolling(100).max()
    roll_low  = df["low"].rolling(100).min()
    df["range100"]     = roll_high - roll_low
    df["pos_in_range"] = (df["close"] - roll_low) / df["range100"]
    ret = df["close"].pct_change()
    df["vol_z"] = (ret - ret.rolling(50).mean()) / ret.rolling(50).std()

    # Swing distance (normalized)
    df["swing_high_20"]   = df["high"].rolling(20).max()
    df["swing_low_20"]    = df["low"].rolling(20).min()
    df["dist_swing_high"] = (df["swing_high_20"] - df["close"]) / df["atr"]
    df["dist_swing_low"]  = (df["close"] - df["swing_low_20"])  / df["atr"]

    # Momentum consistency
    df["mom3"]  = df["candle_dir"].rolling(3).sum()
    df["mom5"]  = df["candle_dir"].rolling(5).sum()
    df["mom10"] = df["candle_dir"].rolling(10).sum()

    df["ret3_atr"]  = (df["close"] - df["close"].shift(3))  / df["atr"]
    df["ret5_atr"]  = (df["close"] - df["close"].shift(5))  / df["atr"]
    df["ret10_atr"] = (df["close"] - df["close"].shift(10)) / df["atr"]

    # Time
    h = df["time"].dt.hour
    df["hour_sin"]    = np.sin(2 * np.pi * h / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * h / 24)
    df["day_of_week"] = df["time"].dt.dayofweek

    return df.dropna().sort_values("time").reset_index(drop=True)


def apply_labels(df, pair):
    pip_mult = 0.001 if "JPY" in pair else 0.00001

    # ── Filter momentum LONGGAR ──────────────────────────────
    # Hanya 3 syarat keras: body cukup besar, wick tidak dominan, arah jelas
    # Semua kondisi konteks (EMA, RSI, volume) jadi FITUR ML, bukan filter
    momentum_base = (
        (df["body_atr"]         >= BODY_ATR_MIN)  &
        (df["upper_wick_body"]  <= WICK_BODY_MAX) &
        (df["lower_wick_body"]  <= WICK_BODY_MAX) &
        (df["body_range_ratio"] >= BODY_RANGE_MIN)
    )

    # Arah candle
    is_bull = df["close"] > df["open"]
    is_bear = df["close"] < df["open"]

    # Filter minimal volume — satu-satunya konteks yang dipertahankan
    # karena candle momentum tanpa volume = false signal yang jelas
    has_volume = df["vol_ratio"] > 0.7  # longgar: 70% dari rata-rata

    df["cand_long"]          = momentum_base & is_bull & has_volume
    df["cand_short"]         = momentum_base & is_bear & has_volume
    df["momentum_candidate"] = df["cand_long"] | df["cand_short"]

    closes    = df["close"].values
    highs     = df["high"].values
    lows      = df["low"].values
    atrs      = df["atr"].values
    spreads   = df["spread"].values
    c_long    = df["cand_long"].values
    c_short   = df["cand_short"].values
    labels    = np.zeros(len(df), dtype=int)
    directions= np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if pd.isna(atrs[i]): continue
        if not (c_long[i] or c_short[i]): continue

        spread_dec = spreads[i] * pip_mult
        sl_dist    = atrs[i] * ATR_SL_MULT
        end_idx    = min(len(df), i + 1 + MAX_HOLD)

        if c_long[i]:
            entry = closes[i] + spread_dec
            sl_p  = entry - sl_dist
            tp_p  = entry + sl_dist * RR_RATIO
            hit   = 0
            for j in range(i+1, end_idx):
                if lows[j]  <= sl_p: break
                if highs[j] >= tp_p: hit = 1; break
            labels[i]     = hit
            directions[i] = 1
        else:
            entry = closes[i] - spread_dec
            sl_p  = entry + sl_dist
            tp_p  = entry - sl_dist * RR_RATIO
            hit   = 0
            for j in range(i+1, end_idx):
                if highs[j] >= sl_p: break
                if lows[j]  <= tp_p: hit = 1; break
            labels[i]     = hit
            directions[i] = -1

    df["label"]     = labels
    df["direction"] = directions
    return df

for p in pairs:
    t0 = time.time()
    try:
        df = pd.read_csv(f"data/{p}/raw.csv")
        df["pair"] = p
        df = build_features(df, p)
        df = apply_labels(df, p)
        df.to_csv(f"data/{p}/labeled.csv", index=False)

        cands  = df["momentum_candidate"].sum()
        n_long = df["cand_long"].sum()
        n_sht  = df["cand_short"].sum()
        wr     = (df["label"].sum() / cands * 100) if cands > 0 else 0
        wr_l   = (df.loc[df["cand_long"],  "label"].sum() / n_long * 100) if n_long > 0 else 0
        wr_s   = (df.loc[df["cand_short"], "label"].sum() / n_sht  * 100) if n_sht  > 0 else 0
        print(f"[{p}] {time.time()-t0:.1f}s | "
              f"Sinyal: {cands:,} (L:{n_long} WR:{wr_l:.1f}% | S:{n_sht} WR:{wr_s:.1f}%) | "
              f"WR total: {wr:.1f}%")
    except FileNotFoundError:
        print(f"Data {p} tidak ditemukan. Skip.")