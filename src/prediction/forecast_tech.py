import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR


class TechForecaster:
    """
    Real-time predictor (single-run, no rolling loop):
    - Balanced core: latest 4 years
    - Aggressive core: latest 3 years
    - Combine target exposure with 50/50 capital split
    """

    balanced_years = 4
    aggressive_years = 3

    def __init__(self, ticker="NVDA", current_exposure_pct=None):
        self.ticker = str(ticker).upper()
        self.file_path = RAW_DIR / f"{self.ticker}_history.csv"
        if current_exposure_pct is None:
            env_val = os.getenv("CURRENT_EXPOSURE_PCT", "0")
            try:
                current_exposure_pct = float(env_val)
            except ValueError:
                current_exposure_pct = 0.0
        self.current_exposure_pct = float(np.clip(current_exposure_pct, 0.0, 100.0))

    @staticmethod
    def _normalize_price_cols(df):
        out = df.copy()
        out.columns = [c.strip().lower() for c in out.columns]
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"price file missing columns: {missing}")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        return out

    @staticmethod
    def _build_features(df):
        d = df.copy()

        # Trend / relative position
        d["SMA_20"] = d["close"].rolling(20).mean()
        d["SMA_50"] = d["close"].rolling(50).mean()
        d["SMA_200"] = d["close"].rolling(200).mean()
        d["Price_vs_SMA20"] = d["close"] / d["SMA_20"] - 1.0
        d["Price_vs_SMA50"] = d["close"] / d["SMA_50"] - 1.0
        d["Price_vs_SMA200"] = d["close"] / d["SMA_200"] - 1.0

        # RSI(14)
        delta = d["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["MACD"] = ema12 - ema26
        d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
        d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

        # ATR(14)
        tr = pd.concat(
            [
                (d["high"] - d["low"]).abs(),
                (d["high"] - d["close"].shift(1)).abs(),
                (d["low"] - d["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        d["ATR_14"] = tr.rolling(14).mean()
        d["ATR_Pct"] = d["ATR_14"] / d["close"].replace(0, np.nan)

        # Returns / volatility / volume behavior
        d["Return_1D"] = d["close"].pct_change()
        d["Momentum_5D"] = d["close"].pct_change(5)
        d["Volatility_20"] = d["Return_1D"].rolling(20).std()
        vol_mu = d["volume"].rolling(20).mean()
        vol_sigma = d["volume"].rolling(20).std().replace(0, np.nan)
        d["Volume_Z20"] = (d["volume"] - vol_mu) / vol_sigma

        # Next-session label: 1(up) / 0(not up)
        d["next_close"] = d["close"].shift(-1)
        d["Target"] = (d["next_close"] > d["close"]).astype(float)
        return d

    def _fit_and_predict_one_core(self, featured_df, years):
        anchor_date = featured_df.index.max()
        start_date = anchor_date - pd.DateOffset(years=int(years))
        window_df = featured_df.loc[featured_df.index >= start_date].copy()

        feature_cols = [
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "Price_vs_SMA200",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "ATR_Pct",
            "Return_1D",
            "Momentum_5D",
            "Volatility_20",
            "Volume_Z20",
        ]

        today_x = window_df.iloc[[-1]][feature_cols].copy()

        # Exclude today's row: tomorrow label unknown in real trading.
        train_df = window_df.iloc[:-1].copy()
        train_df = train_df.dropna(subset=feature_cols + ["Target"]).copy()
        if len(train_df) < 120:
            raise ValueError(
                f"not enough rows after feature/label cleanup in {years}-year core "
                f"(rows={len(train_df)})"
            )

        X = train_df[feature_cols]
        y = train_df["Target"].astype(int)

        if y.nunique() < 2:
            raise ValueError(f"training target has only one class in {years}-year core")

        model = RandomForestClassifier(
            n_estimators=220,
            class_weight="balanced_subsample",
            max_depth=4,
            min_samples_leaf=0.05,
            random_state=42 + years,
            n_jobs=-1,
        )
        model.fit(X, y)

        up_prob = float(model.predict_proba(today_x)[0, 1])
        signal = int(up_prob >= 0.50)
        return {
            "years": int(years),
            "start_date": pd.Timestamp(start_date).date(),
            "anchor_date": pd.Timestamp(anchor_date).date(),
            "up_prob": up_prob,
            "signal": signal,
            "today_features": today_x.iloc[0].to_dict(),
            "train_rows": int(len(train_df)),
        }

    def run_forecast(self):
        print("\n" + "=" * 72)
        print(f" Predictor | {self.ticker} | Real-time 4Y+3Y Dual-Core")
        print("=" * 72)

        if not os.path.exists(self.file_path):
            print(f"Error: data file not found: {self.file_path}")
            print("Run price collection first.")
            return

        raw = pd.read_csv(self.file_path)
        price_df = self._normalize_price_cols(raw)
        featured = self._build_features(price_df)
        featured = featured.replace([np.inf, -np.inf], np.nan)

        if len(featured) < 260:
            print("Error: not enough price history to run predictor.")
            return

        anchor_date = featured.index.max().date()
        next_day = (pd.Timestamp(anchor_date) + pd.Timedelta(days=1)).date()

        print("\n[Step 1] Feature Engineering / Data Ready")
        print(f"Anchor close date : {anchor_date}")
        print(f"Prediction target : next session ({next_day})")

        latest = featured.iloc[-1]
        print("Today key indicators:")
        print(f"  Close      : {latest['close']:.2f}")
        print(f"  RSI(14)    : {latest['RSI_14']:.2f}")
        print(f"  MACD Hist  : {latest['MACD_Hist']:.4f}")
        print(f"  ATR(14)%   : {latest['ATR_Pct'] * 100:.2f}%")

        print("\n[Step 2] Dual-Core Train + Predict")
        bal = self._fit_and_predict_one_core(featured, self.balanced_years)
        agg = self._fit_and_predict_one_core(featured, self.aggressive_years)

        def _sig_text(sig):
            return "LONG(1)" if sig == 1 else "FLAT(0)"

        print(
            f"Balanced core ({bal['years']}Y): "
            f"train={bal['start_date']}~{bal['anchor_date']} "
            f"rows={bal['train_rows']} | up_prob={bal['up_prob']:.3f} | signal={_sig_text(bal['signal'])}"
        )
        print(
            f"  Raw probs (balanced): P(up)={bal['up_prob']*100:.2f}% | "
            f"P(down)={(1.0 - bal['up_prob'])*100:.2f}%"
        )
        print(
            f"Aggressive core ({agg['years']}Y): "
            f"train={agg['start_date']}~{agg['anchor_date']} "
            f"rows={agg['train_rows']} | up_prob={agg['up_prob']:.3f} | signal={_sig_text(agg['signal'])}"
        )
        print(
            f"  Raw probs (aggressive): P(up)={agg['up_prob']*100:.2f}% | "
            f"P(down)={(1.0 - agg['up_prob'])*100:.2f}%"
        )
        combo_up_prob = 0.5 * bal["up_prob"] + 0.5 * agg["up_prob"]
        combo_signal = int(combo_up_prob >= 0.50)
        print(
            "Combo core (50/50 prob blend): "
            f"up_prob={combo_up_prob:.3f} | signal={_sig_text(combo_signal)}"
        )
        print(
            f"  Raw probs (combo): P(up)={combo_up_prob*100:.2f}% | "
            f"P(down)={(1.0 - combo_up_prob)*100:.2f}%"
        )

        print("\n[Step 3] 50/50 Exposure Synthesis")
        balanced_alloc = 50.0 if bal["signal"] == 1 else 0.0
        aggressive_alloc = 50.0 if agg["signal"] == 1 else 0.0
        target_exposure = balanced_alloc + aggressive_alloc
        print(f"Balanced contribution  : {balanced_alloc:.1f}%")
        print(f"Aggressive contribution: {aggressive_alloc:.1f}%")
        print(f"System target exposure : {target_exposure:.1f}%")

        print("\n[Step 4] Human Trading Instruction (for next session)")
        current = float(self.current_exposure_pct)
        diff = target_exposure - current
        print(f"Current actual exposure: {current:.1f}%")
        print(f"Target exposure        : {target_exposure:.1f}%")
        print(f"Required adjustment    : {diff:+.1f}%")

        if abs(diff) < 1e-6:
            action = "No change. Hold current position."
        elif diff > 0:
            action = f"Increase long exposure by {abs(diff):.1f}% at next session open."
        else:
            action = f"Reduce long exposure by {abs(diff):.1f}% at next session open."
        print(f"Instruction            : {action}")
        print("=" * 72 + "\n")


def run_tech_forecast(ticker="NVDA", current_exposure_pct=None):
    forecaster = TechForecaster(ticker=ticker, current_exposure_pct=current_exposure_pct)
    forecaster.run_forecast()


if __name__ == "__main__":
    run_tech_forecast("NVDA")
