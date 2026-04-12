import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import backtesting.backtesting as btb
from backtesting import Backtest, Strategy

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.strategy.backtester1 import backtester1, DEFAULT_COMMISSION, DEFAULT_SPREAD


WINDOWS_DEFAULT = [
    ("2024-01-01", None),
]

TICKER_ALIASES = {
    "APPLE": "AAPL",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "FACEBOOK": "META",
}


class ComboMLStrategy(Strategy):
    max_gross_leverage = 3.0
    rebalance_tolerance = 0.05
    min_target_exposure = 0.03

    def init(self):
        pass

    def _current_signed_exposure(self):
        if not self.position:
            return 0.0
        close = float(self.data.Close[-1])
        eq = float(self.equity)
        if close <= 0 or eq <= 0:
            return 0.0
        notional = float(self.position.size) * close
        return float(np.clip(notional / eq, -self.max_gross_leverage, self.max_gross_leverage))

    def next(self):
        target = float(np.clip(self.data.Target_Exposure[-1], -self.max_gross_leverage, self.max_gross_leverage))

        if abs(target) < self.min_target_exposure:
            if self.position:
                self.position.close()
            return

        current = self._current_signed_exposure()

        # Opposite direction target: flatten first, then reopen on the next bar.
        if self.position and current * target < 0:
            self.position.close()
            return

        # Same direction downsize.
        if self.position and np.sign(current) == np.sign(target) and abs(target) < max(0.0, abs(current) - self.rebalance_tolerance):
            reduce_portion = float(np.clip((abs(current) - abs(target)) / max(abs(current), 1e-9), 0.05, 1.0))
            self.position.close(portion=reduce_portion)
            return

        # Same direction or flat -> add position towards target.
        gap = target - current
        if abs(gap) <= self.rebalance_tolerance:
            return
        add_size = float(np.clip(abs(gap) / max(self.max_gross_leverage, 1e-9), 0.01, 0.99))

        if gap > 0:
            self.buy(size=add_size)
        else:
            self.sell(size=add_size)


class ComboNAVStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy(size=0.999)


def _run_one(ticker, params, rolling_train_years, win_start, win_end, panel_tickers):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            st = backtester1(
                ticker=ticker,
                optimize_params=False,
                anti_overfit_mode=True,
                noise_test=False,
                manual_params=params,
                eval_start=win_start,
                eval_end=win_end,
                train_window_mode="rolling",
                rolling_train_years=int(rolling_train_years),
                train_start="2014-01-01",
                enable_panel_training=True,
                panel_tickers=panel_tickers,
                save_report=False,
                print_summary=False,
                attach_combo_summary=False,
                return_backtest_data=True,
            )
    return st


def _signed_exposure_from_stats(stats_obj):
    backtest_df = stats_obj.get("_backtest_data", pd.DataFrame()).copy()
    if backtest_df.empty or "Close" not in backtest_df.columns:
        return pd.Series(dtype=float)

    idx = backtest_df.index
    close = pd.to_numeric(backtest_df["Close"], errors="coerce").reindex(idx)
    eq_curve = stats_obj.get("_equity_curve", pd.DataFrame())
    if isinstance(eq_curve, pd.DataFrame) and "Equity" in eq_curve.columns:
        eq = pd.to_numeric(eq_curve["Equity"], errors="coerce").reindex(idx).ffill().bfill()
    else:
        eq = pd.Series(np.nan, index=idx)
    trades = stats_obj.get("_trades", pd.DataFrame())

    units = pd.Series(0.0, index=idx, dtype=float)
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        for _, tr in trades.iterrows():
            try:
                size = float(tr.get("Size", 0.0))
                et = pd.to_datetime(tr.get("EntryTime"), errors="coerce")
                xt = pd.to_datetime(tr.get("ExitTime"), errors="coerce")
                if not np.isfinite(size) or size == 0.0 or pd.isna(et) or pd.isna(xt):
                    continue
                mask = (idx >= et) & (idx <= xt)
                if mask.any():
                    units.loc[mask] += size
            except Exception:
                continue

    notional = units * close
    exposure = (notional / eq.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return exposure.clip(-3.0, 3.0)


def _build_combo_window_frame(st_bal, st_agg, wb, wa):
    bal_df = st_bal.get("_backtest_data", pd.DataFrame()).copy()
    agg_df = st_agg.get("_backtest_data", pd.DataFrame()).copy()
    if bal_df.empty or agg_df.empty:
        return pd.DataFrame()

    required_ohlc = ["Open", "High", "Low", "Close"]
    for col in required_ohlc:
        if col not in bal_df.columns:
            return pd.DataFrame()

    bal_frame = bal_df[required_ohlc + ["ATR_20"]].copy()
    frame = bal_frame.join(
        pd.DataFrame(index=agg_df.index, data={"_agg_exists": 1}),
        how="inner"
    ).drop(columns=["_agg_exists"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.replace([np.inf, -np.inf], np.nan)

    if "ATR_20" not in frame.columns or frame["ATR_20"].isna().all():
        tr = pd.concat(
            [
                (frame["High"] - frame["Low"]).abs(),
                (frame["High"] - frame["Close"].shift(1)).abs(),
                (frame["Low"] - frame["Close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["ATR_20"] = tr.rolling(20, min_periods=5).mean()

    bal_exp = _signed_exposure_from_stats(st_bal).reindex(frame.index).ffill().fillna(0.0)
    agg_exp = _signed_exposure_from_stats(st_agg).reindex(frame.index).ffill().fillna(0.0)
    frame["Exposure_Balanced"] = bal_exp
    frame["Exposure_Aggressive"] = agg_exp
    frame["Target_Exposure"] = (wb * bal_exp + wa * agg_exp).clip(-3.0, 3.0)

    frame = frame.dropna(subset=required_ohlc + ["Target_Exposure"]).copy()
    return frame


def _selected_params_from_stats(stats_obj):
    if stats_obj is None:
        return {}
    raw = stats_obj.get("_selected_params", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _build_eval_span(windows):
    starts = []
    ends = []
    for ws, we in windows:
        ts = pd.to_datetime(ws, errors="coerce")
        if pd.notna(ts):
            starts.append(ts)
        te = pd.to_datetime(we, errors="coerce") if we is not None else pd.NaT
        if pd.notna(te):
            ends.append(te)
    eval_start = min(starts).strftime("%Y-%m-%d") if starts else "2024-01-01"
    eval_end = max(ends).strftime("%Y-%m-%d") if ends else None
    return eval_start, eval_end


def _auto_generate_param_file(
    ticker,
    out_path,
    rolling_train_years,
    windows,
    panel_tickers,
    optimize_samples=600,
    optimize_seed=123,
):
    eval_start, eval_end = _build_eval_span(windows)
    print(
        f"[Combo] Auto-generating params for {ticker} "
        f"(rolling={rolling_train_years}y, eval={eval_start}->{eval_end or 'latest'}, "
        f"samples={optimize_samples})..."
    )
    st = backtester1(
        ticker=ticker,
        optimize_params=True,
        optimize_samples=int(max(50, optimize_samples)),
        optimize_seed=int(optimize_seed + rolling_train_years),
        anti_overfit_mode=True,
        noise_test=False,
        eval_start=eval_start,
        eval_end=eval_end,
        train_window_mode="rolling",
        rolling_train_years=int(rolling_train_years),
        train_start="2014-01-01",
        enable_panel_training=True,
        panel_tickers=panel_tickers,
        save_report=False,
        print_summary=False,
        attach_combo_summary=False,
    )
    params = _selected_params_from_stats(st)
    if not params:
        raise RuntimeError(
            f"Auto-generation failed for {ticker}: optimizer did not return selected params."
        )

    payload = {
        "ticker": str(ticker).upper(),
        "rolling_train_years": int(rolling_train_years),
        "params": params,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "generator": "combo_backtest_report.auto",
        "eval_start": eval_start,
        "eval_end": eval_end,
        "optimize_samples": int(max(50, optimize_samples)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[Combo] Saved ticker-specific params: {out_path}")
    return out_path


def run_combo_report(
    ticker="NVDA",
    balanced_params_path=None,
    aggressive_params_path=None,
    windows=None,
    combo_weight_balanced=0.5,
    panel_tickers=("NVDA", "AMD", "TSM"),
    combo_engine="live_allocator",
    auto_generate_params=True,
    optimize_samples_per_core=600,
    optimize_seed=123,
):
    raw_ticker = str(ticker).upper()
    ticker = TICKER_ALIASES.get(raw_ticker, raw_ticker)
    if ticker != raw_ticker:
        print(f"[Combo] Ticker alias mapped: {raw_ticker} -> {ticker}")
    windows = windows or WINDOWS_DEFAULT

    required_raw = Path("data/raw") / f"{ticker}_history.csv"
    required_merged = Path("data/processed") / f"{ticker}_sentiment_stock_merged.csv"
    missing_inputs = [str(p) for p in [required_raw, required_merged] if not p.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            f"Missing required data for {ticker}: {missing_inputs}. "
            "Please run step 1 (Collect data) and step 2 (Run sentiment analysis) first."
        )
    if balanced_params_path is None:
        balanced_params_path = Path("data/processed") / f"{ticker}_three_window_best_params.json"
    else:
        balanced_params_path = Path(balanced_params_path)
    if aggressive_params_path is None:
        aggressive_params_path = Path("data/processed") / f"{ticker}_three_window_best_params_v2.json"
    else:
        aggressive_params_path = Path(aggressive_params_path)

    if auto_generate_params and not balanced_params_path.exists():
        _auto_generate_param_file(
            ticker=ticker,
            out_path=balanced_params_path,
            rolling_train_years=4,
            windows=windows,
            panel_tickers=panel_tickers,
            optimize_samples=optimize_samples_per_core,
            optimize_seed=optimize_seed,
        )
    if auto_generate_params and not aggressive_params_path.exists():
        _auto_generate_param_file(
            ticker=ticker,
            out_path=aggressive_params_path,
            rolling_train_years=3,
            windows=windows,
            panel_tickers=panel_tickers,
            optimize_samples=optimize_samples_per_core,
            optimize_seed=optimize_seed + 17,
        )

    if not balanced_params_path.exists():
        raise FileNotFoundError(
            f"Missing balanced params file: {balanced_params_path}. "
            "Enable auto_generate_params or generate ticker-specific params first."
        )
    if not aggressive_params_path.exists():
        raise FileNotFoundError(
            f"Missing aggressive params file: {aggressive_params_path}. "
            "Enable auto_generate_params or generate ticker-specific params first."
        )

    balanced_cfg = json.loads(balanced_params_path.read_text())
    aggressive_cfg = json.loads(aggressive_params_path.read_text())

    wb = float(combo_weight_balanced)
    wb = min(max(wb, 0.0), 1.0)
    wa = 1.0 - wb
    combo_engine = str(combo_engine).lower().strip()
    if combo_engine not in {"sleeve_nav", "live_allocator"}:
        raise ValueError("combo_engine must be one of: sleeve_nav, live_allocator")

    btb._tqdm = lambda seq, *args, **kwargs: seq
    combo_segments = []
    combo_equity_segments = []
    real_close_segments = []
    sleeve_bal_trades = 0
    sleeve_agg_trades = 0
    sleeve_bal_commission = 0.0
    sleeve_agg_commission = 0.0

    for win_start, win_end in windows:
        st_bal = _run_one(
            ticker=ticker,
            params=balanced_cfg["params"],
            rolling_train_years=balanced_cfg["rolling_train_years"],
            win_start=win_start,
            win_end=win_end,
            panel_tickers=panel_tickers,
        )
        st_agg = _run_one(
            ticker=ticker,
            params=aggressive_cfg["params"],
            rolling_train_years=aggressive_cfg["rolling_train_years"],
            win_start=win_start,
            win_end=win_end,
            panel_tickers=panel_tickers,
        )
        if st_bal is None or st_agg is None:
            continue
        real_close = pd.to_numeric(
            st_bal.get("_backtest_data", pd.DataFrame()).get("Close"),
            errors="coerce"
        )
        if real_close is not None:
            real_close_segments.append(real_close.rename("Close"))
        sleeve_bal_trades += int(float(st_bal.get("# Trades", 0)))
        sleeve_agg_trades += int(float(st_agg.get("# Trades", 0)))
        sleeve_bal_commission += float(st_bal.get("Commissions [$]", 0.0))
        sleeve_agg_commission += float(st_agg.get("Commissions [$]", 0.0))
        ec_bal = st_bal.get("_equity_curve", pd.DataFrame()).get("Equity")
        ec_agg = st_agg.get("_equity_curve", pd.DataFrame()).get("Equity")
        if ec_bal is not None and ec_agg is not None:
            curve = pd.concat(
                {
                    "Equity_Balanced": pd.to_numeric(ec_bal, errors="coerce"),
                    "Equity_Aggressive": pd.to_numeric(ec_agg, errors="coerce"),
                },
                axis=1,
            ).sort_index().ffill().bfill()
            curve["Equity_Combo_50_50"] = wb * curve["Equity_Balanced"] + wa * curve["Equity_Aggressive"]
            combo_equity_segments.append(curve["Equity_Combo_50_50"].rename("Close"))
            curve_out = Path("data/processed") / f"{ticker}_window_{win_start}_{win_end or 'latest'}_equity_compare.csv"
            curve.to_csv(curve_out, index_label="Date")

        if combo_engine == "live_allocator":
            window_frame = _build_combo_window_frame(st_bal, st_agg, wb=wb, wa=wa)
            if not window_frame.empty:
                combo_segments.append(window_frame)

    if combo_engine == "live_allocator":
        if not combo_segments:
            raise RuntimeError("No valid combo segments built from windows.")
        combo_data = pd.concat(combo_segments, axis=0).sort_index()
        combo_data = combo_data[~combo_data.index.duplicated(keep="first")]
        combo_data = combo_data.dropna(subset=["Open", "High", "Low", "Close", "Target_Exposure"])
        combo_data_out = Path("data/processed") / f"{ticker}_combo_merged_backtest_input.csv"
        combo_data.to_csv(combo_data_out, index_label="Date")

        bt = Backtest(
            combo_data,
            ComboMLStrategy,
            cash=10000,
            commission=DEFAULT_COMMISSION,
            spread=DEFAULT_SPREAD,
            margin=1.0 / ComboMLStrategy.max_gross_leverage,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        close_source = combo_data["Close"]
    else:
        if not combo_equity_segments:
            raise RuntimeError("No valid sleeve equity curves built from windows.")
        combo_close = pd.concat(combo_equity_segments, axis=0).sort_index()
        combo_close = combo_close[~combo_close.index.duplicated(keep="first")]
        combo_close = pd.to_numeric(combo_close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if combo_close.empty:
            raise RuntimeError("Combined sleeve equity curve is empty.")

        combo_price = combo_close / float(combo_close.iloc[0]) * 100.0
        combo_data = pd.DataFrame(index=combo_close.index)
        combo_data["Close"] = combo_price
        combo_data["Open"] = combo_data["Close"].shift(1).fillna(combo_data["Close"])
        combo_data["High"] = np.maximum(combo_data["Open"], combo_data["Close"])
        combo_data["Low"] = np.minimum(combo_data["Open"], combo_data["Close"])
        combo_data["Volume"] = 1.0
        combo_data_out = Path("data/processed") / f"{ticker}_combo_sleeve_nav_input.csv"
        combo_data.to_csv(combo_data_out, index_label="Date")

        bt = Backtest(
            combo_data,
            ComboNAVStrategy,
            cash=10000.0,
            commission=0.0,
            spread=0.0,
            trade_on_close=True,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        close_source = combo_data["Close"]

    bh_return_pct = np.nan
    bh_max_dd_pct = np.nan
    if real_close_segments:
        close_vals = pd.concat(real_close_segments, axis=0).sort_index()
        close_vals = close_vals[~close_vals.index.duplicated(keep="first")]
    else:
        close_vals = pd.to_numeric(close_source, errors="coerce")
    close_vals = pd.to_numeric(close_vals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    close_vals = close_vals[close_vals > 0]
    if not close_vals.empty:
        bh_curve = close_vals / float(close_vals.iloc[0])
        bh_return_pct = float((bh_curve.iloc[-1] - 1.0) * 100.0)
        bh_dd = (bh_curve / bh_curve.cummax()) - 1.0
        bh_max_dd_pct = float(bh_dd.min() * 100.0)
    stats["Buy & Hold Return [%]"] = bh_return_pct
    stats["Buy & Hold Max. Drawdown [%]"] = bh_max_dd_pct
    stats["Combo Weight Balanced"] = wb
    stats["Combo Weight Aggressive"] = wa
    stats["Combo Mode"] = (
        "sleeve_nav_from_real_substrategy_equity"
        if combo_engine == "sleeve_nav"
        else "live_blend_from_real_sleeve_exposure"
    )
    stats["Combo Input Rows"] = int(len(combo_data))
    stats["Combo Input Start"] = combo_data.index.min()
    stats["Combo Input End"] = combo_data.index.max()
    stats["Sleeve Balanced # Trades"] = sleeve_bal_trades
    stats["Sleeve Aggressive # Trades"] = sleeve_agg_trades
    stats["Sleeve Combined # Trades"] = sleeve_bal_trades + sleeve_agg_trades
    stats["Sleeve Balanced Commissions [$]"] = sleeve_bal_commission
    stats["Sleeve Aggressive Commissions [$]"] = sleeve_agg_commission
    stats["Sleeve Combined Commissions [$]"] = sleeve_bal_commission + sleeve_agg_commission

    print("\nBacktest summary (Combo 50/50)")
    print("-" * 40)
    print(stats)

    html_path = Path("data/processed") / f"{ticker}_COMBO_ML_Final.html"
    bt.plot(filename=str(html_path), open_browser=False)
    print(f"\nMerged combo input saved to: {combo_data_out}")
    print(f"Report saved to: {html_path}")
    return stats


if __name__ == "__main__":
    run_combo_report("NVDA")
