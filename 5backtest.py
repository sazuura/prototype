"""
6backtest.py
============
Backtest final untuk skripsi.
Baca prediksi dari 5train.py → simulasi eksekusi → equity curve.

Output:
  data/backtest_trades.csv    ← detail setiap trade
  data/backtest_report.csv    ← ringkasan per tahun
  data/equity_curve.csv       ← equity curve harian (untuk chart)
"""

import pandas as pd
import numpy as np
import os

# ══════════════════════════════════════════════════════════════
# PARAMETER TERKUNCI
# ══════════════════════════════════════════════════════════════
PRED_PATH      = "data/train_predictions.csv"
INITIAL_BAL    = 1000.0
RISK_PERCENT   = 0.01       # 1% risk per trade
RR_RATIO       = 1.0
ATR_SL_MULT    = 2.0
MAX_HOLD       = 40         # bar M15 = 10 jam
CONTRACT_SIZE  = 100_000
MAX_LOT        = 10.0
USE_BE         = False      # break-even off — hasil bersih
SESSION_START  = 7          # UTC — London open
SESSION_END    = 17         # UTC — NY close
VOL_REGIME_MIN = 0.8
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("  BACKTEST FINAL — Rolling Yearly | Random Forest")
print("=" * 65)

# ── Load prediksi ─────────────────────────────────────────────
if not os.path.exists(PRED_PATH):
    print(f"File {PRED_PATH} tidak ditemukan.")
    print("Jalankan dulu: python 5train.py")
    exit()

preds = pd.read_csv(PRED_PATH, parse_dates=["time"])
preds = preds.sort_values("time").reset_index(drop=True)

# Ambil hanya sinyal yang melewati threshold
signals = preds[preds["pred"] == 1].copy()
print(f"\nTotal sinyal   : {len(signals):,}")
print(f"Periode        : {signals['time'].min().date()} → {signals['time'].max().date()}")
print(f"Pairs          : {signals['pair'].unique().tolist()}")

# ── Load raw data per pair ────────────────────────────────────
pairs     = signals["pair"].unique()
raw_cache = {}
for p in pairs:
    path = f"data/{p}/labeled.csv"
    if os.path.exists(path):
        raw_cache[p] = pd.read_csv(path, parse_dates=["time"])
    else:
        print(f"  WARN: {path} tidak ditemukan")


# ── Lot sizing ────────────────────────────────────────────────
def calc_lot(balance, risk_pct, sl_dist, pair, price):
    risk_usd = balance * risk_pct
    if sl_dist <= 0: return 0.01
    pip_mult = 0.01 if "JPY" in pair else 0.0001
    sl_pips  = sl_dist / pip_mult
    if pair.endswith("USD"):
        pip_val = 10.0
    elif pair.startswith("USD"):
        pip_val = (pip_mult / price) * CONTRACT_SIZE
    else:
        pip_val = 10.0
    lot = risk_usd / (sl_pips * pip_val)
    return min(MAX_LOT, max(0.01, round(lot, 2)))


def calc_pnl(entry, exit_p, lot, pair, direction):
    diff = (exit_p - entry) * direction
    if pair.endswith("USD"):
        return diff * lot * CONTRACT_SIZE
    elif pair.startswith("USD"):
        return (diff / exit_p) * lot * CONTRACT_SIZE
    return diff * lot * CONTRACT_SIZE


# ── Simulasi trade ────────────────────────────────────────────
balance = INITIAL_BAL
trades  = []

for _, sig in signals.iterrows():
    pair   = sig["pair"]
    t_in   = sig["time"]
    direct = int(sig.get("direction", 1))

    df_p = raw_cache.get(pair)
    if df_p is None: continue

    # Filter sesi trading
    if not (SESSION_START <= t_in.hour < SESSION_END):
        continue

    # Cari bar entry
    match = df_p[df_p["time"] == t_in]
    if match.empty: continue
    idx = match.index[0]
    bar = df_p.iloc[idx]

    # Filter regime volatilitas
    vol_regime = bar.get("vol_regime", 1.0)
    if pd.notna(vol_regime) and float(vol_regime) < VOL_REGIME_MIN:
        continue

    pip_m   = 0.001 if "JPY" in pair else 0.00001
    sp_dec  = float(bar["spread"]) * pip_m
    sl_dist = float(bar["atr"]) * ATR_SL_MULT

    if direct == 1:
        entry = float(bar["close"]) + sp_dec
        sl_p  = entry - sl_dist
        tp_p  = entry + sl_dist * RR_RATIO
    else:
        entry = float(bar["close"]) - sp_dec
        sl_p  = entry + sl_dist
        tp_p  = entry - sl_dist * RR_RATIO

    lot    = calc_lot(balance, RISK_PERCENT, sl_dist, pair, entry)
    future = df_p.iloc[idx+1 : idx+1+MAX_HOLD]

    status = "Timeout"
    exit_p = future.iloc[-1]["close"] if not future.empty else entry
    exit_t = future.iloc[-1]["time"]  if not future.empty else t_in

    for _, fb in future.iterrows():
        if direct == 1:
            if fb["low"]  <= sl_p: status="SL"; exit_p=sl_p; exit_t=fb["time"]; break
            if fb["high"] >= tp_p: status="TP"; exit_p=tp_p; exit_t=fb["time"]; break
        else:
            if fb["high"] >= sl_p: status="SL"; exit_p=sl_p; exit_t=fb["time"]; break
            if fb["low"]  <= tp_p: status="TP"; exit_p=tp_p; exit_t=fb["time"]; break

    pnl = calc_pnl(entry, exit_p, lot, pair, direct)
    if status == "Timeout":
        status = "Timeout(+)" if pnl > 0 else "Timeout(-)"

    balance += pnl
    trades.append({
        "year"     : t_in.year,
        "time_in"  : t_in,
        "time_out" : exit_t,
        "pair"     : pair,
        "direction": "LONG" if direct==1 else "SHORT",
        "status"   : status,
        "lot"      : lot,
        "entry"    : round(entry, 5),
        "exit"     : round(float(exit_p), 5),
        "sl"       : round(sl_p, 5),
        "tp"       : round(float(tp_p), 5),
        "pnl"      : round(pnl, 2),
        "balance"  : round(balance, 2),
    })

# ── Analytics ─────────────────────────────────────────────────
if not trades:
    print("\nTidak ada trade tereksekusi.")
    print("Cek filter sesi / vol_regime / threshold.")
    exit()

res = pd.DataFrame(trades)
res["win"]  = (res["pnl"] > 0).astype(int)
res["loss"] = (res["pnl"] <= 0).astype(int)

# Drawdown
res["peak"]    = res["balance"].cummax()
res["dd_usd"]  = res["peak"] - res["balance"]
res["dd_pct"]  = res["dd_usd"] / res["peak"] * 100
max_dd_usd     = res["dd_usd"].max()
max_dd_pct     = res["dd_pct"].max()

# Streaks
res["w_streak"] = res["win"].groupby(
    (res["win"] != res["win"].shift()).cumsum()).cumsum()
res["l_streak"] = res["loss"].groupby(
    (res["loss"] != res["loss"].shift()).cumsum()).cumsum()

total_pnl      = balance - INITIAL_BAL
win_rate       = res["win"].mean() * 100
n_years        = (res["time_in"].max() - res["time_in"].min()).days / 365.25
cagr           = ((balance / INITIAL_BAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
rec_factor     = total_pnl / max_dd_usd if max_dd_usd > 0 else 0

# Sharpe
res["date"]    = pd.to_datetime(res["time_in"]).dt.date
daily_pnl      = res.groupby("date")["pnl"].sum()
sharpe         = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

# Profit factor
gross_win      = res.loc[res["win"]==1, "pnl"].sum()
gross_loss     = res.loc[res["win"]==0, "pnl"].abs().sum()
profit_factor  = gross_win / gross_loss if gross_loss > 0 else np.inf

# ── Report ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'  OVERALL PERFORMANCE REPORT  ':=^65}")
print(f"Net Profit       : ${total_pnl:,.2f} ({total_pnl/INITIAL_BAL*100:.2f}%)")
print(f"CAGR             : {cagr:.2f}% / tahun")
print(f"Max Drawdown     : ${max_dd_usd:.2f} ({max_dd_pct:.2f}%)")
print(f"Recovery Factor  : {rec_factor:.2f}")
print(f"Sharpe Ratio     : {sharpe:.2f}")
print(f"Profit Factor    : {profit_factor:.2f}")
print(f"Win Rate         : {win_rate:.2f}%")
print(f"Total Trades     : {len(res)}")
print(f"Max Win Streak   : {res['w_streak'].max()}")
print(f"Max Loss Streak  : {res['l_streak'].max()}")
print("-" * 65)
print("Status breakdown:")
print(res["status"].value_counts().to_string())

# ── Annual breakdown ──────────────────────────────────────────
yearly = res.groupby("year").agg(
    Trades = ("pnl", "count"),
    Profit = ("pnl", "sum"),
    WR     = ("win", "mean"),
).copy()
yearly["WR"]     = (yearly["WR"] * 100).round(2).astype(str) + "%"
yearly["Profit"] = yearly["Profit"].map("${:,.2f}".format)

print("\n" + "=" * 65)
print(f"{'  ANNUAL BREAKDOWN  ':=^65}")
print(yearly.to_string())

# ── Per pair ──────────────────────────────────────────────────
per_pair = res.groupby("pair").agg(
    Trades = ("pnl", "count"),
    Profit = ("pnl", "sum"),
    WR     = ("win", "mean"),
).sort_values("Profit", ascending=False)
per_pair["WR"]     = (per_pair["WR"] * 100).round(2).astype(str) + "%"
per_pair["Profit"] = per_pair["Profit"].map("${:,.2f}".format)

print("\n" + "=" * 65)
print(f"{'  PER PAIR BREAKDOWN  ':=^65}")
print(per_pair.to_string())

# ── Equity curve harian ───────────────────────────────────────
eq = res.groupby("date").agg(
    pnl     = ("pnl", "sum"),
    balance = ("balance", "last"),
    trades  = ("pnl", "count"),
).reset_index()
eq["cumulative_return"] = (eq["balance"] / INITIAL_BAL - 1) * 100

# ── Simpan ────────────────────────────────────────────────────
res_out = res.drop(columns=["win","loss","peak","dd_usd","dd_pct",
                              "w_streak","l_streak","date"], errors="ignore")
res_out.to_csv("data/backtest_trades.csv", index=False)
eq.to_csv("data/equity_curve.csv", index=False)

report_rows = []
for yr, grp in res.groupby("year"):
    yr_pnl = grp["pnl"].sum()
    yr_wr  = grp["win"].mean()
    report_rows.append({
        "year": yr, "trades": len(grp),
        "profit": round(yr_pnl, 2),
        "win_rate": round(yr_wr, 4),
        "profit_pct": round(yr_pnl / INITIAL_BAL * 100, 2),
    })
pd.DataFrame(report_rows).to_csv("data/backtest_report.csv", index=False)

print("\n" + "=" * 65)
print(f"{'  KONFIGURASI  ':=^65}")
print(f"Risk/trade       : {RISK_PERCENT*100:.0f}%")
print(f"RR Ratio         : 1:{RR_RATIO}")
print(f"SL Multiplier    : {ATR_SL_MULT}x ATR")
print(f"Max Hold         : {MAX_HOLD} bar ({MAX_HOLD*15//60} jam)")
print(f"Session Filter   : {SESSION_START}:00–{SESSION_END}:00 UTC")
print(f"Vol Regime Min   : {VOL_REGIME_MIN}")
print(f"Break-Even       : {'ON' if USE_BE else 'OFF'}")
print(f"Spread           : Dihitung dinamis")
print("=" * 65)
print("\nFile output:")
print("  data/backtest_trades.csv")
print("  data/backtest_report.csv")
print("  data/equity_curve.csv")