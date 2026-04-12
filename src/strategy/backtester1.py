import os
import json
import contextlib
import pandas as pd
import numpy as np
import backtesting.backtesting as btb
from pathlib import Path
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import time
import sys

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import PROCESSED_DIR, RAW_DIR

DEFAULT_COMMISSION = 0.0001
DEFAULT_SPREAD = 0.0008

# ==========================================
# 2. Trading strategy
# ==========================================
class MLStrategy(Strategy):
    confidence_entry = 0.35
    exit_prob = 0.50  # retained for backward compatibility
    min_exposure = 0.10
    max_exposure = 3.00
    max_gross_leverage = 3.00
    target_vol_annual = 1.20
    rebalance_tolerance = 0.30
    min_size = 0.10  # retained for backward compatibility
    max_size = 0.99  # retained for backward compatibility
    target_daily_risk = 0.015  # retained for backward compatibility
    atr_mult = 4.0  # retained for backward compatibility
    trail_atr_mult = 3.5
    entry_sl_atr_mult = 1.6
    min_trail_atr_mult = 1.0
    time_tighten_start = 15
    time_tighten_step = 5
    time_tighten_factor = 0.98
    tp_mult = 0.0
    time_stop_bars = 12
    loss_cut_pct = 0.07
    max_position_drawdown = 0.12
    profit_lock_activate = 0.05
    max_profit_giveback = 0.30
    drawdown_pause = 0.12
    cooldown_bars = 0
    vix_limit = 20.0
    regime_require_both = False
    exit_on_regime_break = False
    trend_buffer = 0.94
    trend_exit_buffer = 0.94
    prob_lookback = 126
    min_prob_history = 40
    entry_quantile = 0.35
    exit_quantile = 0.20  # retained for backward compatibility
    meta_threshold = 0.50
    finbert_min_prob = 0.25
    finbert_z_min = -0.50
    macro_relaxed_entry = 0.45
    putcall_max = 1.20
    min_peer_rs = 0.00
    adx_entry_min = 25.0
    adx_chop_level = 25.0
    chop_stop_tighten = 0.85
    short_enabled = False
    short_entry_quantile = 0.30
    short_exit_quantile = 0.60
    short_min_size = 0.08
    short_max_size = 0.35
    short_vix_floor = 18.0
    short_trend_buffer = 1.00
    chop_filter_enabled = False
    chop_adx_limit = 18.0
    chop_band_atr = 0.65
    chop_long_prob = 0.20
    chop_short_prob = 0.80
    chop_max_exposure = 0.30
    pyramid_lookback = 20
    pyramid_add_exposure = 0.20
    max_pyramids = 2

    def init(self):
        self.entry_bar = None
        self.equity_peak = self.equity
        self.cooldown_left = 0
        self.dd_pause_armed = False
        self.position_peak_pl_pct = 0.0
        self.pyramid_count = 0
        self.last_pyramid_bar = -1

    def _regime_allows_long(self):
        vix_ok = True
        if hasattr(self.data, 'VIX_Close') and len(self.data.VIX_Close) > 0:
            vix_val = float(self.data.VIX_Close[-1])
            if np.isfinite(vix_val):
                vix_ok = vix_val < self.vix_limit

        spy_ok = True
        if hasattr(self.data, 'SPY_Above_200DMA') and len(self.data.SPY_Above_200DMA) > 0:
            spy_ok = float(self.data.SPY_Above_200DMA[-1]) > 0

        if self.regime_require_both:
            return vix_ok and spy_ok
        return vix_ok or spy_ok

    def _trend_allows_long(self):
        if hasattr(self.data, 'SMA_200') and len(self.data.SMA_200) > 0:
            sma200 = float(self.data.SMA_200[-1])
            close = float(self.data.Close[-1])
            if np.isfinite(sma200) and sma200 > 0:
                return close >= sma200 * self.trend_buffer
        return True

    def _trend_allows_short(self):
        if hasattr(self.data, 'SMA_200') and len(self.data.SMA_200) > 0:
            sma200 = float(self.data.SMA_200[-1])
            close = float(self.data.Close[-1])
            if np.isfinite(sma200) and sma200 > 0:
                return close <= sma200 * self.short_trend_buffer
        return False

    def _regime_allows_short(self):
        vix_ok = False
        if hasattr(self.data, 'VIX_Close') and len(self.data.VIX_Close) > 0:
            vix_val = float(self.data.VIX_Close[-1])
            if np.isfinite(vix_val):
                vix_ok = vix_val >= self.short_vix_floor

        spy_below = False
        if hasattr(self.data, 'SPY_Above_200DMA') and len(self.data.SPY_Above_200DMA) > 0:
            spy_below = float(self.data.SPY_Above_200DMA[-1]) <= 0
        return vix_ok or spy_below

    def _prob_rank(self, prob):
        if len(self.data.Pred_Prob) <= 1:
            return 0.5
        hist = np.asarray(self.data.Pred_Prob[:-1], dtype=float)
        hist = hist[np.isfinite(hist)]
        if len(hist) < self.min_prob_history:
            return 0.5
        lookback = int(max(self.min_prob_history, self.prob_lookback))
        hist = hist[-lookback:]
        return float(np.clip(np.mean(hist <= prob), 0.0, 1.0))

    def _trend_still_intact(self):
        if hasattr(self.data, 'SMA_200') and len(self.data.SMA_200) > 0:
            sma200 = float(self.data.SMA_200[-1])
            close = float(self.data.Close[-1])
            if np.isfinite(sma200) and sma200 > 0:
                return close >= sma200 * self.trend_exit_buffer
        return True

    def _realized_vol_annual(self, close_price, atr_value):
        if hasattr(self.data, 'HV_20_Ann') and len(self.data.HV_20_Ann) > 0:
            hv = float(self.data.HV_20_Ann[-1])
            if np.isfinite(hv) and hv > 0:
                return hv
        if np.isfinite(close_price) and close_price > 0 and np.isfinite(atr_value) and atr_value > 0:
            return (atr_value / close_price) * np.sqrt(252)
        return np.nan

    def _position_size(self, prob, close_price, atr_value):
        """
        Volatility targeting + conviction scaling:
        - High vol -> smaller exposure (toward 20%)
        - Low vol -> larger exposure (up to 150%)
        """
        rank = self._prob_rank(prob)
        realized_vol = self._realized_vol_annual(close_price, atr_value)

        if np.isfinite(realized_vol) and realized_vol > 0:
            vol_exposure = self.target_vol_annual / realized_vol
        else:
            vol_exposure = 0.80

        vol_exposure = float(np.clip(vol_exposure, self.min_exposure, self.max_exposure))

        conviction_scale = 0.70 + 0.80 * rank  # 0.70 -> 1.50
        target_exposure = float(np.clip(
            vol_exposure * conviction_scale,
            self.min_exposure,
            self.max_exposure
        ))

        leverage_cap = max(1.0, self.max_gross_leverage)
        size_fraction = float(np.clip(target_exposure / leverage_cap, 0.01, 0.99))
        return size_fraction

    def _position_size_short(self, prob):
        rank = self._prob_rank(prob)
        bearish_conviction = float(np.clip(1.0 - rank, 0.0, 1.0))
        target_exposure = self.short_min_size + (self.short_max_size - self.short_min_size) * bearish_conviction
        leverage_cap = max(1.0, self.max_gross_leverage)
        return float(np.clip(target_exposure / leverage_cap, 0.01, 0.99))

    def _update_trailing_stop(self, close_price, atr_value, trail_mult):
        if not np.isfinite(atr_value) or atr_value <= 0:
            return
        trail_sl_long = close_price - trail_mult * atr_value
        trail_sl_short = close_price + trail_mult * atr_value
        for trade in self.trades:
            if trade.is_long:
                if np.isfinite(trail_sl_long) and trail_sl_long > 0:
                    if trade.sl is None or trail_sl_long > trade.sl:
                        trade.sl = trail_sl_long
            elif trade.is_short:
                if np.isfinite(trail_sl_short) and trail_sl_short > 0:
                    if trade.sl is None or trail_sl_short < trade.sl:
                        trade.sl = trail_sl_short

    def _current_exposure(self, close_price):
        if not self.position or not np.isfinite(close_price) or close_price <= 0:
            return 0.0
        notional = abs(float(self.position.size) * close_price)
        return float(notional / max(self.equity, 1e-9))

    def _rebalance_long_position(self, prob, close_price, atr_value):
        if not self.position or not self.position.is_long:
            return

        target_size = self._position_size(prob, close_price, atr_value)
        leverage_cap = max(1.0, self.max_gross_leverage)
        target_exposure = float(target_size * leverage_cap)
        current_exposure = self._current_exposure(close_price)
        tol = float(max(0.05, self.rebalance_tolerance))
        max_allowed_exposure = float(max(self.min_exposure, self.max_exposure))

        if current_exposure > max_allowed_exposure * (1.0 + tol):
            reduce_portion = 1.0 - (max_allowed_exposure / max(current_exposure, 1e-9))
            reduce_portion = float(np.clip(reduce_portion, 0.05, 1.0))
            self.position.close(portion=reduce_portion)
            return

        if current_exposure < target_exposure * (1.0 - tol):
            add_exposure = target_exposure - current_exposure
            add_size = float(np.clip(add_exposure / leverage_cap, 0.0, 0.30))
            if add_size >= 0.01:
                self.buy(size=add_size)

    def _rebalance_short_position(self, prob, close_price):
        if not self.position or not self.position.is_short:
            return

        target_size = self._position_size_short(prob)
        leverage_cap = max(1.0, self.max_gross_leverage)
        target_exposure = float(target_size * leverage_cap)
        current_exposure = self._current_exposure(close_price)
        tol = float(max(0.05, self.rebalance_tolerance))
        max_allowed_exposure = float(max(self.short_min_size, self.short_max_size))

        if current_exposure > max_allowed_exposure * (1.0 + tol):
            reduce_portion = 1.0 - (max_allowed_exposure / max(current_exposure, 1e-9))
            reduce_portion = float(np.clip(reduce_portion, 0.05, 1.0))
            self.position.close(portion=reduce_portion)
            return

        if current_exposure < target_exposure * (1.0 - tol):
            add_exposure = target_exposure - current_exposure
            add_size = float(np.clip(add_exposure / leverage_cap, 0.0, 0.25))
            if add_size >= 0.01:
                self.sell(size=add_size)

    def _dynamic_trail_mult(self, held_bars, adx_value):
        trail_mult = float(self.trail_atr_mult)
        if held_bars >= self.time_tighten_start:
            step = max(1, int(self.time_tighten_step))
            tighten_steps = 1 + ((held_bars - self.time_tighten_start) // step)
            trail_mult *= float(self.time_tighten_factor) ** tighten_steps
        if np.isfinite(adx_value) and adx_value < self.adx_chop_level:
            trail_mult *= self.chop_stop_tighten
        return float(np.clip(trail_mult, self.min_trail_atr_mult, self.trail_atr_mult))

    def _maybe_pyramid(self, close_price, prob, finbert_ok):
        if not self.position or not self.position.is_long:
            return
        if self.pyramid_count >= int(max(0, self.max_pyramids)):
            return
        if len(self.data.Close) <= int(self.pyramid_lookback):
            return
        if len(self.data.Close) - 1 == self.last_pyramid_bar:
            return
        if not finbert_ok:
            return

        recent = np.asarray(self.data.Close[-(int(self.pyramid_lookback) + 1):-1], dtype=float)
        recent = recent[np.isfinite(recent)]
        if recent.size == 0:
            return
        breakout_high = float(np.max(recent))
        if close_price <= breakout_high:
            return
        if prob < self.macro_relaxed_entry:
            return

        current_exposure = self._current_exposure(close_price)
        room = float(max(0.0, self.max_exposure - current_exposure))
        if room <= 0.01:
            return

        add_exposure = float(min(self.pyramid_add_exposure, room))
        leverage_cap = max(1.0, self.max_gross_leverage)
        add_size = float(np.clip(add_exposure / leverage_cap, 0.0, 0.30))
        if add_size >= 0.01:
            self.buy(size=add_size)
            self.pyramid_count += 1
            self.last_pyramid_bar = len(self.data.Close) - 1

    def _dynamic_prob_threshold(self, q, fallback):
        if len(self.data.Pred_Prob) <= 1:
            return fallback

        hist = np.asarray(self.data.Pred_Prob[:-1], dtype=float)
        hist = hist[np.isfinite(hist)]
        if len(hist) < self.min_prob_history:
            return fallback

        lookback = int(max(self.min_prob_history, self.prob_lookback))
        hist = hist[-lookback:]
        dyn = float(np.quantile(hist, q))
        if not np.isfinite(dyn):
            return fallback
        return float(np.clip(dyn, 0.01, 0.99))

    def _reset_position_trackers(self):
        self.entry_bar = None
        self.position_peak_pl_pct = 0.0
        self.pyramid_count = 0
        self.last_pyramid_bar = -1

    def next(self):
        if len(self.data.Pred_Prob) == 0:
            return

        prob = float(self.data.Pred_Prob[-1])
        prob_rank = self._prob_rank(prob)
        close_price = float(self.data.Close[-1])
        atr_value = float(self.data.ATR_20[-1]) if len(self.data.ATR_20) > 0 else np.nan
        adx_value = float(self.data.ADX_14[-1]) if hasattr(self.data, 'ADX_14') and len(self.data.ADX_14) > 0 else np.nan
        self.equity_peak = max(self.equity_peak, self.equity)
        dd = 1.0 - (self.equity / max(self.equity_peak, 1e-9))
        entry_threshold = self._dynamic_prob_threshold(self.entry_quantile, self.confidence_entry)
        entry_threshold = min(entry_threshold, self.confidence_entry)
        short_threshold = self._dynamic_prob_threshold(self.short_entry_quantile, 1.0 - self.confidence_entry)
        qqq_above_50 = float(self.data.QQQ_Above_50DMA[-1]) if hasattr(self.data, 'QQQ_Above_50DMA') and len(self.data.QQQ_Above_50DMA) > 0 else np.nan
        if np.isfinite(qqq_above_50) and qqq_above_50 > 0:
            entry_threshold = min(entry_threshold, self.macro_relaxed_entry)
        meta_prob = float(self.data.Meta_Prob[-1]) if hasattr(self.data, 'Meta_Prob') and len(self.data.Meta_Prob) > 0 else 1.0
        meta_ok = np.isfinite(meta_prob) and (meta_prob >= self.meta_threshold)
        adx_ok = (not np.isfinite(adx_value)) or (adx_value >= self.adx_entry_min)
        is_chop = np.isfinite(adx_value) and (adx_value < self.chop_adx_limit)
        finbert_prob = float(self.data.FinBERT_Prob[-1]) if hasattr(self.data, 'FinBERT_Prob') and len(self.data.FinBERT_Prob) > 0 else np.nan
        finbert_z = float(self.data.FinBERT_Prob_Z20[-1]) if hasattr(self.data, 'FinBERT_Prob_Z20') and len(self.data.FinBERT_Prob_Z20) > 0 else np.nan
        finbert_ok = (
            (np.isfinite(finbert_z) and finbert_z >= self.finbert_z_min)
            or (np.isfinite(finbert_prob) and finbert_prob >= self.finbert_min_prob)
            or (not np.isfinite(finbert_z) and not np.isfinite(finbert_prob))
        )
        put_call_total = float(self.data.PutCall_Total[-1]) if hasattr(self.data, 'PutCall_Total') and len(self.data.PutCall_Total) > 0 else np.nan
        put_call_ok = (not np.isfinite(put_call_total)) or (put_call_total <= self.putcall_max)

        if not self.position and self.entry_bar is not None:
            self._reset_position_trackers()

        if self.position:
            # Model only handles entry. Exit is managed by trend integrity + ATR trailing stop.
            held_bars = len(self.data.Close) - self.entry_bar if self.entry_bar is not None else 0
            current_pl_pct = float(self.position.pl_pct)
            self.position_peak_pl_pct = max(self.position_peak_pl_pct, current_pl_pct)

            if dd >= self.max_position_drawdown:
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = max(self.cooldown_bars, 5)
                self.dd_pause_armed = True
                return

            if (
                self.position_peak_pl_pct >= self.profit_lock_activate
                and current_pl_pct <= (self.position_peak_pl_pct * (1.0 - self.max_profit_giveback))
            ):
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return

            if held_bars >= int(max(1, self.time_stop_bars)) and current_pl_pct < 0:
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return

            if self.position.is_long and current_pl_pct <= -abs(self.loss_cut_pct):
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return
            if self.position.is_short and current_pl_pct <= -abs(self.loss_cut_pct):
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return

            if (
                self.position.is_long
                and self.exit_on_regime_break
                and (not self._regime_allows_long())
                and current_pl_pct <= 0
            ):
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return
            if (
                self.position.is_short
                and self.exit_on_regime_break
                and (not self._regime_allows_short())
                and current_pl_pct <= 0
            ):
                self.position.close()
                self._reset_position_trackers()
                self.cooldown_left = self.cooldown_bars
                return

            trail_mult = self._dynamic_trail_mult(held_bars, adx_value)
            self._update_trailing_stop(close_price, atr_value, trail_mult)
            if self.position.is_long:
                self._maybe_pyramid(close_price, prob, finbert_ok)
                self._rebalance_long_position(prob, close_price, atr_value)
                if not self._trend_still_intact():
                    self.position.close()
                    self._reset_position_trackers()
                    self.cooldown_left = self.cooldown_bars
            elif self.position.is_short:
                self._rebalance_short_position(prob, close_price)
                if (not self._trend_allows_short()) and current_pl_pct <= 0:
                    self.position.close()
                    self._reset_position_trackers()
                    self.cooldown_left = self.cooldown_bars
            return

        if not self.position:
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
                return

            # Drawdown circuit breaker: pause new entries after deep equity pullback.
            if dd >= self.drawdown_pause and not self.dd_pause_armed:
                self.cooldown_left = max(self.cooldown_bars, 5)
                self.dd_pause_armed = True
                return
            if self.dd_pause_armed and dd < (self.drawdown_pause * 0.5):
                self.dd_pause_armed = False
            if self.cooldown_left > 0:
                return

            if self.chop_filter_enabled and is_chop and np.isfinite(atr_value) and atr_value > 0:
                sma20 = float(self.data.SMA_20[-1]) if hasattr(self.data, 'SMA_20') and len(self.data.SMA_20) > 0 else np.nan
                if np.isfinite(sma20):
                    deviation = (close_price - sma20) / atr_value
                    chop_size_cap = self.chop_max_exposure / max(1.0, self.max_gross_leverage)
                    if (
                        deviation <= -self.chop_band_atr
                        and prob_rank >= self.chop_long_prob
                        and meta_ok
                        and put_call_ok
                        and self._regime_allows_long()
                    ):
                        size = min(self._position_size(prob, close_price, atr_value), chop_size_cap)
                        sl_price = close_price - max(0.8, self.entry_sl_atr_mult) * atr_value
                        tp_price = sma20 if sma20 > close_price else (close_price + 0.5 * atr_value)
                        self.buy(size=size, sl=sl_price, tp=tp_price)
                        self.entry_bar = len(self.data.Close)
                        self.position_peak_pl_pct = 0.0
                        self.pyramid_count = 0
                        self.last_pyramid_bar = self.entry_bar - 1
                        return
                    if (
                        self.short_enabled
                        and deviation >= self.chop_band_atr
                        and prob_rank <= self.chop_short_prob
                        and meta_ok
                        and self._regime_allows_short()
                    ):
                        size = min(self._position_size_short(prob), chop_size_cap)
                        sl_price = close_price + max(0.8, self.entry_sl_atr_mult) * atr_value
                        tp_price = sma20 if sma20 < close_price else (close_price - 0.5 * atr_value)
                        self.sell(size=size, sl=sl_price, tp=tp_price)
                        self.entry_bar = len(self.data.Close)
                        self.position_peak_pl_pct = 0.0
                        self.pyramid_count = 0
                        self.last_pyramid_bar = self.entry_bar - 1
                        return
                return

            can_long = False
            can_short = False

            if prob > entry_threshold and meta_ok and adx_ok and finbert_ok and put_call_ok:
                can_long = self._regime_allows_long() and self._trend_allows_long()
                if can_long and hasattr(self.data, 'Peer_RS_20D') and len(self.data.Peer_RS_20D) > 0:
                    peer_rs = float(self.data.Peer_RS_20D[-1])
                    if np.isfinite(peer_rs) and peer_rs < self.min_peer_rs:
                        can_long = False
                if can_long and hasattr(self.data, 'RS_vs_QQQ_20D') and len(self.data.RS_vs_QQQ_20D) > 0:
                    qqq_rs = float(self.data.RS_vs_QQQ_20D[-1])
                    if np.isfinite(qqq_rs) and qqq_rs < -0.04:
                        can_long = False
                if can_long and hasattr(self.data, 'RS_vs_SOXX_20D') and len(self.data.RS_vs_SOXX_20D) > 0:
                    soxx_rs = float(self.data.RS_vs_SOXX_20D[-1])
                    if np.isfinite(soxx_rs) and soxx_rs < -0.04:
                        can_long = False
                if can_long and hasattr(self.data, 'CMF_20') and len(self.data.CMF_20) > 0:
                    cmf = float(self.data.CMF_20[-1])
                    if np.isfinite(cmf) and cmf < -0.05:
                        can_long = False

            if self.short_enabled and prob < short_threshold and meta_ok and adx_ok:
                can_short = self._regime_allows_short() and self._trend_allows_short()
                if can_short and np.isfinite(finbert_z) and finbert_z > 1.0:
                    can_short = False
                if can_short and hasattr(self.data, 'RS_vs_QQQ_20D') and len(self.data.RS_vs_QQQ_20D) > 0:
                    qqq_rs = float(self.data.RS_vs_QQQ_20D[-1])
                    if np.isfinite(qqq_rs) and qqq_rs > 0.05:
                        can_short = False

            if can_long and can_short:
                long_edge = prob - entry_threshold
                short_edge = short_threshold - prob
                if short_edge > long_edge:
                    can_long = False
                else:
                    can_short = False

            if can_long:
                size = self._position_size(prob, close_price, atr_value)
                sl_price = None
                if np.isfinite(atr_value) and atr_value > 0:
                    sl_price = close_price - self.entry_sl_atr_mult * atr_value
                    if sl_price <= 0:
                        sl_price = None
                self.buy(size=size, sl=sl_price, tp=None)
                self.entry_bar = len(self.data.Close)
                self.position_peak_pl_pct = 0.0
                self.pyramid_count = 0
                self.last_pyramid_bar = self.entry_bar - 1
                return

            if can_short:
                size = self._position_size_short(prob)
                sl_price = None
                if np.isfinite(atr_value) and atr_value > 0:
                    sl_price = close_price + self.entry_sl_atr_mult * atr_value
                self.sell(size=size, sl=sl_price, tp=None)
                self.entry_bar = len(self.data.Close)
                self.position_peak_pl_pct = 0.0
                self.pyramid_count = 0
                self.last_pyramid_bar = self.entry_bar - 1

# ==========================================
# 3. Core backtest function
# ==========================================
def backtester1(
    ticker="META",
    optimize_params=False,
    optimize_samples=5000,
    optimize_seed=123,
    anti_overfit_mode=True,
    noise_test=True,
    noise_level=0.001,
    noise_runs=5,
    purge_bars=10,
    manual_params=None,
    enable_panel_training=True,
    panel_tickers=("NVDA", "AMD", "TSM"),
    eval_start=None,
    eval_end=None,
    train_window_mode="expanding",
    rolling_train_years=5,
    train_start="2014-01-01",
    save_report=True,
    print_summary=True,
    attach_combo_summary=True,
    combo_weight_balanced=0.5,
    return_backtest_data=False,
):
    print(f"Starting machine-learning backtest for {ticker}...")
    ticker = str(ticker).upper()
    regime_start = pd.to_datetime(train_start, errors='coerce')
    if pd.isna(regime_start):
        regime_start = pd.Timestamp("2014-01-01")
    regime_start = pd.Timestamp(regime_start).tz_localize(None)
    train_window_mode = str(train_window_mode).lower().strip()
    if train_window_mode not in {"single", "expanding", "rolling"}:
        print("Error: train_window_mode must be one of: single, expanding, rolling.")
        return
    print(
        "Generalization mode: cross-sectional normalized features + panel training "
        f"({'enabled' if enable_panel_training else 'disabled'})."
    )
    print(
        f"Training window mode: {train_window_mode} | "
        f"feature/train start: {regime_start.date()}"
    )

    def _roll_z(series, window=20):
        s = pd.to_numeric(series, errors='coerce')
        mu = s.rolling(window=window, min_periods=max(5, window // 4)).mean()
        sigma = s.rolling(window=window, min_periods=max(5, window // 4)).std().replace(0, np.nan)
        out = (s - mu) / sigma
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)

    def _prepare_ticker_frame(symbol):
        symbol = str(symbol).upper()
        csv_path = PROCESSED_DIR / f"{symbol}_sentiment_stock_merged.csv"
        price_path = RAW_DIR / f"{symbol}_history.csv"
        if not os.path.exists(csv_path) or not os.path.exists(price_path):
            return None

        feature_df = pd.read_csv(csv_path)
        if 'date' not in feature_df.columns:
            return None
        feature_df['date'] = pd.to_datetime(feature_df['date'], errors='coerce').dt.tz_localize(None)
        feature_df = feature_df.dropna(subset=['date']).sort_values('date')

        price_df = pd.read_csv(price_path)
        if 'Date' not in price_df.columns:
            return None
        price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce').dt.tz_localize(None)
        price_df = price_df.dropna(subset=['Date']).sort_values('Date')

        frame = pd.merge(
            price_df,
            feature_df,
            left_on='Date',
            right_on='date',
            how='inner'
        )
        frame = frame.drop(columns=['date'], errors='ignore')
        if 'lexicon_score' in frame.columns:
            frame = frame.drop(columns=['lexicon_score'])
        if 'Close_x' in frame.columns:
            frame = frame.rename(columns={'Close_x': 'Close'})
        if 'Close_y' in frame.columns:
            frame = frame.drop(columns=['Close_y'])
        frame = frame[frame['Date'] >= regime_start].copy()
        if frame.empty:
            return None
        frame = frame.set_index('Date')

        frame['Sentiment'] = pd.to_numeric(frame.get('adaptive_score', 0.0), errors='coerce').fillna(0.0)
        frame['Sent_Z20'] = _roll_z(frame['Sentiment'], window=20)
        frame['Sent_5D_Chg'] = frame['Sentiment'].diff(5)
        frame['Sent_5D_Chg_Z20'] = _roll_z(frame['Sent_5D_Chg'], window=20)
        frame['Sent_Abs'] = frame['Sentiment'].abs()

        frame['Close'] = pd.to_numeric(frame['Close'], errors='coerce')
        frame['Returns'] = frame['Close'].pct_change(fill_method=None)
        frame['Returns_Z20'] = _roll_z(frame['Returns'], window=20)
        frame['SMA_5'] = frame['Close'].rolling(window=5).mean()
        frame['SMA_20'] = frame['Close'].rolling(window=20).mean()
        frame['SMA_50'] = frame['Close'].rolling(window=50).mean()
        frame['SMA_200'] = frame['Close'].rolling(window=200).mean()
        frame['Vol_5'] = frame['Close'].rolling(window=5).std()
        frame['Vol5_Pct'] = frame['Vol_5'] / frame['Close'].replace(0, np.nan)
        frame['Vol5_Pct_Z20'] = _roll_z(frame['Vol5_Pct'], window=20)
        frame['HV_20_Ann'] = frame['Returns'].rolling(window=20).std() * np.sqrt(252)

        tr_components = pd.concat(
            [
                (frame['High'] - frame['Low']).abs(),
                (frame['High'] - frame['Close'].shift(1)).abs(),
                (frame['Low'] - frame['Close'].shift(1)).abs(),
            ],
            axis=1
        )
        tr14 = tr_components.max(axis=1)
        frame['ATR_20'] = tr14.rolling(window=20).mean()
        frame['ATR_Pct'] = frame['ATR_20'] / frame['Close'].replace(0, np.nan)
        frame['ATR_Pct_Z20'] = _roll_z(frame['ATR_Pct'], window=20)
        up_move = frame['High'].diff()
        down_move = -frame['Low'].diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=frame.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=frame.index
        )
        atr14 = tr14.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        frame['ADX_14'] = dx.ewm(alpha=1 / 14, adjust=False).mean()
        frame['ADX_Z20'] = _roll_z(frame['ADX_14'], window=20)

        fast_ema = frame['Close'].ewm(span=12, adjust=False).mean()
        slow_ema = frame['Close'].ewm(span=26, adjust=False).mean()
        frame['MACD'] = fast_ema - slow_ema
        frame['MACD_Z20'] = _roll_z(frame['MACD'], window=20)

        delta = frame['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        frame['RSI_14'] = 100 - (100 / (1 + rs))
        frame['RSI_Z20'] = _roll_z(frame['RSI_14'], window=20)

        frame['Price_vs_SMA20'] = frame['Close'] / frame['SMA_20'].replace(0, np.nan) - 1.0
        frame['Price_vs_SMA50'] = frame['Close'] / frame['SMA_50'].replace(0, np.nan) - 1.0
        frame['Price_vs_SMA200'] = frame['Close'] / frame['SMA_200'].replace(0, np.nan) - 1.0
        frame['SMA20_vs_SMA50'] = frame['SMA_20'] / frame['SMA_50'].replace(0, np.nan) - 1.0

        if 'VIX_Close' in frame.columns:
            frame['VIX_Close'] = pd.to_numeric(frame['VIX_Close'], errors='coerce')
            frame['VIX_Rel20'] = (
                frame['VIX_Close'] / frame['VIX_Close'].rolling(20, min_periods=5).mean().replace(0, np.nan) - 1.0
            )
        else:
            frame['VIX_Rel20'] = 0.0

        for c in ['Peer_RS_20D', 'RS_vs_QQQ_20D', 'RS_vs_SOXX_20D', 'SupplyChain_Sent', 'SC_TSM_Sent', 'SC_ASML_Sent', 'SC_SMCI_Sent']:
            if c in frame.columns:
                frame[f'{c}_Z20'] = _roll_z(frame[c], window=20)

        news_recency_src = (
            frame['News_Recency_Days']
            if 'News_Recency_Days' in frame.columns
            else pd.Series(999.0, index=frame.index)
        )
        news_recency_days = pd.to_numeric(news_recency_src, errors='coerce').reindex(frame.index).fillna(999.0)
        frame['News_Recency_Decay'] = np.exp(-news_recency_days / 10.0)

        if 'Target' not in frame.columns:
            return None
        frame['Target'] = pd.to_numeric(frame['Target'], errors='coerce')
        frame = frame.dropna(subset=['Target']).copy()
        frame['Target'] = frame['Target'].astype(int)
        frame['Symbol'] = symbol
        return frame

    def _cross_sectional_normalize(frames_by_symbol, base_cols):
        if len(frames_by_symbol) < 2:
            return
        rel_source_cols = ['RSI_14', 'ATR_Pct', 'Momentum_5D']
        stacked_frames = []
        for sym, sym_df in frames_by_symbol.items():
            local = sym_df.copy()
            for col in base_cols + rel_source_cols:
                if col not in local.columns:
                    local[col] = np.nan
                local[col] = pd.to_numeric(local[col], errors='coerce')
            local = local[base_cols + rel_source_cols].copy()
            local['Date'] = local.index
            local['Symbol'] = sym
            stacked_frames.append(local.reset_index(drop=True))
        panel = pd.concat(stacked_frames, axis=0, ignore_index=True)
        for col in base_cols:
            grp = panel.groupby('Date')[col]
            mu = grp.transform('mean')
            sigma = grp.transform('std').replace(0, np.nan)
            panel[f'{col}_CSZ'] = ((panel[col] - mu) / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)
        # Cross-sectional relative-strength features ("who is leader today?")
        panel['Relative_RSI'] = (
            (pd.to_numeric(panel['RSI_14'], errors='coerce') - panel.groupby('Date')['RSI_14'].transform('mean')) / 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 2)
        rsi_sigma = panel.groupby('Date')['RSI_14'].transform('std').replace(0, np.nan)
        panel['Relative_RSI_Z'] = (
            (pd.to_numeric(panel['RSI_14'], errors='coerce') - panel.groupby('Date')['RSI_14'].transform('mean')) / rsi_sigma
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)

        panel['Relative_ATR_Pct'] = (
            pd.to_numeric(panel['ATR_Pct'], errors='coerce') - panel.groupby('Date')['ATR_Pct'].transform('mean')
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 2)
        atr_sigma = panel.groupby('Date')['ATR_Pct'].transform('std').replace(0, np.nan)
        panel['Relative_ATR_Pct_Z'] = (
            (pd.to_numeric(panel['ATR_Pct'], errors='coerce') - panel.groupby('Date')['ATR_Pct'].transform('mean')) / atr_sigma
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)

        panel['Relative_Momentum_5D_Z'] = (
            (
                pd.to_numeric(panel['Momentum_5D'], errors='coerce')
                - panel.groupby('Date')['Momentum_5D'].transform('mean')
            )
            / panel.groupby('Date')['Momentum_5D'].transform('std').replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)
        keep_cols = ['Date', 'Symbol'] + [f'{c}_CSZ' for c in base_cols]
        keep_cols += ['Relative_RSI', 'Relative_RSI_Z', 'Relative_ATR_Pct', 'Relative_ATR_Pct_Z', 'Relative_Momentum_5D_Z']
        panel = panel[keep_cols]
        for sym, sym_df in frames_by_symbol.items():
            cs = panel.loc[panel['Symbol'] == sym].set_index('Date')
            for col in base_cols:
                csz_col = f'{col}_CSZ'
                sym_df[csz_col] = cs[csz_col].reindex(sym_df.index).fillna(0.0)
                base_src = sym_df[col] if col in sym_df.columns else pd.Series(0.0, index=sym_df.index)
                base_col = pd.to_numeric(base_src, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # Blend TS-normalized signal with cross-sectional z-score to reduce small-panel noise.
                sym_df[f'{col}_GN'] = (0.70 * base_col + 0.30 * sym_df[csz_col]).clip(-8, 8)
            for rel_col in ['Relative_RSI', 'Relative_RSI_Z', 'Relative_ATR_Pct', 'Relative_ATR_Pct_Z', 'Relative_Momentum_5D_Z']:
                sym_df[rel_col] = cs[rel_col].reindex(sym_df.index).fillna(0.0)
            frames_by_symbol[sym] = sym_df

    requested_panel = [ticker]
    if enable_panel_training:
        requested_panel.extend([str(s).upper() for s in panel_tickers])
    requested_panel = list(dict.fromkeys(requested_panel))

    frames_by_symbol = {}
    skipped_symbols = []
    for sym in requested_panel:
        frame = _prepare_ticker_frame(sym)
        if frame is None or frame.empty or len(frame) < 260:
            skipped_symbols.append(sym)
            continue
        frames_by_symbol[sym] = frame

    if ticker not in frames_by_symbol:
        print(f"Error: unable to build training frame for {ticker}. Ensure merged and price files exist.")
        return

    panel_symbols_used = [s for s in requested_panel if s in frames_by_symbol]
    panel_aux_symbols = [s for s in panel_symbols_used if s != ticker]
    panel_training_active = enable_panel_training and len(panel_aux_symbols) > 0

    if enable_panel_training and skipped_symbols:
        print(f"Panel warning: skipped symbols due missing/short data: {skipped_symbols}")
    if panel_training_active:
        print(f"Panel training symbols: {panel_symbols_used}")
    elif enable_panel_training:
        print("Panel warning: no auxiliary symbols available; using single-ticker training.")

    cross_sectional_cols = [
        'FinBERT_Prob_Z20', 'FinBERT_Prob_5D_Chg', 'FinBERT_Conviction',
        'Sent_Z20', 'Sent_5D_Chg_Z20', 'sentiment_impulse', 'sentiment_decay', 'news_intensity_z20',
        'Price_vs_SMA20', 'Price_vs_SMA50', 'Price_vs_SMA200', 'SMA20_vs_SMA50',
        'Returns_Z20', 'Vol5_Pct_Z20', 'ATR_Pct_Z20', 'MACD_Z20', 'RSI_Z20', 'ADX_Z20',
        'Peer_RS_20D_Z20', 'RS_vs_QQQ_20D_Z20', 'RS_vs_SOXX_20D_Z20',
        'VIX_Z20', 'VIX_Rel20', 'PutCall_Z20',
        'SupplyChain_Sent_Z20', 'SC_TSM_Sent_Z20', 'SC_ASML_Sent_Z20', 'SC_SMCI_Sent_Z20'
    ]
    if panel_training_active:
        _cross_sectional_normalize(frames_by_symbol, cross_sectional_cols)

    df = frames_by_symbol[ticker].copy()
    print(f"Dataset ready: {len(df)} trading sessions")

    # --- C. Pruned, normalized feature set for cross-ticker generalization ---
    print("\nTraining model...")
    suffix = "_GN" if panel_training_active else ""
    core_feature_candidates = [
        f'FinBERT_Prob_Z20{suffix}',
        f'Sent_Z20{suffix}',
        f'Sent_5D_Chg_Z20{suffix}',
        f'news_intensity_z20{suffix}',
        'News_Recency_Decay',
        f'Price_vs_SMA50{suffix}',
        f'Price_vs_SMA200{suffix}',
        f'Returns_Z20{suffix}',
        f'ATR_Pct_Z20{suffix}',
        f'MACD_Z20{suffix}',
        f'RSI_Z20{suffix}',
        f'ADX_Z20{suffix}',
        f'RS_vs_QQQ_20D_Z20{suffix}',
        f'Peer_RS_20D_Z20{suffix}',
        f'VIX_Rel20{suffix}',
        f'PutCall_Z20{suffix}',
        'QQQ_Above_50DMA',
        'SPY_Above_200DMA',
        'Risk_On'
    ]
    if panel_training_active:
        core_feature_candidates.extend([
            'Relative_RSI', 'Relative_RSI_Z',
            'Relative_ATR_Pct', 'Relative_ATR_Pct_Z',
            'Relative_Momentum_5D_Z'
        ])
    extended_feature_candidates = [
        f'SupplyChain_Sent_Z20{suffix}', f'SC_TSM_Sent_Z20{suffix}', f'SC_ASML_Sent_Z20{suffix}', f'SC_SMCI_Sent_Z20{suffix}'
    ]
    feature_candidates = core_feature_candidates + ([] if anti_overfit_mode else extended_feature_candidates)
    feature_cols = [c for c in feature_candidates if c in df.columns]
    forbidden_absolute = {
        'Close', 'Open', 'High', 'Low',
        'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
        'ATR_20', 'MACD', 'RSI_14', 'VIX_Close',
        'PutCall_Total', 'PutCall_Equity', 'PutCall_Index'
    }
    feature_cols = [c for c in feature_cols if c not in forbidden_absolute]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    low_var_cols = []
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty or vals.nunique() <= 1:
            low_var_cols.append(col)
    if low_var_cols:
        feature_cols = [c for c in feature_cols if c not in low_var_cols]
        print(f"Pruned low-variance features: {len(low_var_cols)}")
    if not feature_cols:
        print("Error: no valid normalized feature columns found.")
        return

    def _sanitize_feature_frame(frame):
        clean = frame.copy()
        for col in feature_cols:
            if col not in clean.columns:
                clean[col] = 0.0
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
        clean[feature_cols] = clean[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clean['Target'] = pd.to_numeric(clean['Target'], errors='coerce')
        clean = clean.dropna(subset=['Target']).copy()
        clean['Target'] = clean['Target'].astype(int)
        return clean

    def _build_panel_train_frame(target_slice, cutoff_date, start_date=None):
        local_target = target_slice.copy()
        if start_date is not None:
            start_date = pd.Timestamp(start_date).tz_localize(None)
            local_target = local_target.loc[local_target.index >= start_date].copy()
        blocks = [local_target]
        if panel_training_active:
            for sym in panel_aux_symbols:
                aux = frames_by_symbol[sym]
                aux_slice = aux.loc[aux.index <= cutoff_date].copy()
                if start_date is not None:
                    aux_slice = aux_slice.loc[aux_slice.index >= start_date].copy()
                if len(aux_slice) >= 120:
                    blocks.append(aux_slice)
        merged = pd.concat(blocks, axis=0).sort_index()
        return _sanitize_feature_frame(merged)

    # --- D. Anti-overfit split + probability generation ---
    forced_window = (eval_start is not None) or (eval_end is not None)
    if forced_window:
        start_ts = pd.to_datetime(eval_start) if eval_start is not None else df.index.min()
        end_ts = pd.to_datetime(eval_end) if eval_end is not None else df.index.max()
        if pd.isna(start_ts) or pd.isna(end_ts):
            print("Error: invalid eval_start/eval_end.")
            return
        start_ts = pd.Timestamp(start_ts).tz_localize(None)
        end_ts = pd.Timestamp(end_ts).tz_localize(None)
        if start_ts > end_ts:
            print("Error: eval_start must be <= eval_end.")
            return
        holdout_mask = (df.index >= start_ts) & (df.index <= end_ts)
        dev_mask = df.index < start_ts
        print(
            f"\nForced eval window: {start_ts.date()} -> {end_ts.date()} "
            f"(dev rows={int(dev_mask.sum())}, eval rows={int(holdout_mask.sum())})"
        )
    elif anti_overfit_mode:
        unique_years = sorted(pd.Index(df.index.year).unique().tolist())
        if len(unique_years) >= 5:
            holdout_years = unique_years[-2:]
            holdout_mask = df.index.year.isin(holdout_years)
            dev_mask = ~holdout_mask
            print(
                f"\nAnti-overfit split: dev years={unique_years[:-2]}, "
                f"holdout years={holdout_years}"
            )
        else:
            split = int(len(df) * 0.7)
            dev_mask = pd.Series(False, index=df.index)
            dev_mask.iloc[:split] = True
            holdout_mask = ~dev_mask
            print("\nAnti-overfit fallback: insufficient year buckets, using 70/30 time split.")
    else:
        split = int(len(df) * 0.7)
        dev_mask = pd.Series(False, index=df.index)
        dev_mask.iloc[:split] = True
        holdout_mask = ~dev_mask

    dev_df = df.loc[dev_mask].copy()
    holdout_df = df.loc[holdout_mask].copy()
    if forced_window:
        if len(dev_df) < 180 or len(holdout_df) < 80:
            print(
                "Error: forced eval window has too little data. "
                f"dev={len(dev_df)}, eval={len(holdout_df)}"
            )
            return
    elif len(dev_df) < 400 or len(holdout_df) < 120:
        split = int(len(df) * 0.7)
        dev_df = df.iloc[:split].copy()
        holdout_df = df.iloc[split:].copy()
        print("\nAnti-overfit fallback: using 70/30 split due small buckets.")

    purge_bars = int(max(0, purge_bars))
    dev_fit_df = dev_df.copy()
    if anti_overfit_mode and purge_bars > 0 and len(dev_df) > (300 + purge_bars):
        dev_fit_df = dev_df.iloc[:-purge_bars].copy()
        print(
            f"Applied purge gap: removed last {purge_bars} bars from development set "
            "before final model fitting."
        )

    dev_fit_df = _sanitize_feature_frame(dev_fit_df)
    holdout_df = _sanitize_feature_frame(holdout_df)
    min_train_rows = 180

    tech_priority = [
        f'Price_vs_SMA50{suffix}', f'Price_vs_SMA200{suffix}',
        f'Returns_Z20{suffix}', f'ATR_Pct_Z20{suffix}',
        f'MACD_Z20{suffix}', f'RSI_Z20{suffix}', f'ADX_Z20{suffix}',
        f'RS_vs_QQQ_20D_Z20{suffix}', f'Peer_RS_20D_Z20{suffix}',
        f'VIX_Rel20{suffix}', f'PutCall_Z20{suffix}',
        'QQQ_Above_50DMA', 'SPY_Above_200DMA', 'Risk_On'
    ]
    sent_priority = [
        f'FinBERT_Prob_Z20{suffix}', f'Sent_Z20{suffix}',
        f'Sent_5D_Chg_Z20{suffix}', f'news_intensity_z20{suffix}',
        'News_Recency_Decay',
        f'sentiment_impulse{suffix}', f'sentiment_decay{suffix}'
    ]
    if panel_training_active:
        tech_priority.extend(['Relative_RSI_Z', 'Relative_ATR_Pct_Z', 'Relative_Momentum_5D_Z'])
    tech_cols = [c for c in tech_priority if c in feature_cols]
    sent_cols = [c for c in sent_priority if c in feature_cols]
    if len(tech_cols) < 4:
        tech_cols = feature_cols.copy()
    if len(sent_cols) < 3:
        sent_cols = tech_cols.copy()

    def _build_sample_weight(frame):
        n = len(frame)
        if n <= 1:
            return np.ones(n, dtype=float)
        w = np.linspace(1.0, 1.8, n)
        adx_col = f'ADX_Z20{suffix}' if f'ADX_Z20{suffix}' in frame.columns else ('ADX_Z20' if 'ADX_Z20' in frame.columns else None)
        if adx_col is not None:
            adx_z = pd.to_numeric(frame[adx_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w *= np.where(adx_z < -0.5, 1.10, 1.0)
        vix_col = f'VIX_Rel20{suffix}' if f'VIX_Rel20{suffix}' in frame.columns else ('VIX_Rel20' if 'VIX_Rel20' in frame.columns else None)
        if vix_col is not None:
            vix_rel = pd.to_numeric(frame[vix_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w *= np.where(vix_rel < 0.15, 1.08, 0.95)
        atr_col = f'ATR_Pct_Z20{suffix}' if f'ATR_Pct_Z20{suffix}' in frame.columns else ('ATR_Pct_Z20' if 'ATR_Pct_Z20' in frame.columns else None)
        if atr_col is not None:
            atr_z = pd.to_numeric(frame[atr_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w *= np.where(np.abs(atr_z) < 1.5, 1.05, 0.92)
        if 'SPY_Above_200DMA' in frame.columns:
            spy = pd.to_numeric(frame['SPY_Above_200DMA'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
            w *= np.where(spy > 0.0, 1.08, 0.95)
        if 'Target_Quality' in frame.columns:
            tq = pd.to_numeric(frame['Target_Quality'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
            tq = np.clip(tq, 0.5, 2.5)
            w *= tq
        if 'News_Recency_Days' in frame.columns:
            rec = pd.to_numeric(frame['News_Recency_Days'], errors='coerce').fillna(999.0).to_numpy(dtype=float)
            # When news is very stale, down-weight those observations.
            w *= (0.80 + 0.20 * np.exp(-rec / 10.0))
        return np.clip(w, 0.5, 5.0)

    ensemble_seed_offsets = [0, 23]

    def _fit_rf_ensemble(train_df, cols, target, seed):
        models = []
        sample_weight = _build_sample_weight(train_df)
        for offset in ensemble_seed_offsets:
            rf = RandomForestClassifier(
                n_estimators=160,
                class_weight='balanced_subsample',
                max_depth=3,
                min_samples_split=0.08,
                min_samples_leaf=0.08,
                random_state=seed + offset,
                n_jobs=-1
            )
            rf.fit(train_df[cols], target, sample_weight=sample_weight)
            models.append(rf)
        return models

    def _predict_rf_ensemble(models, frame, cols):
        probs = [m.predict_proba(frame[cols])[:, 1] for m in models]
        return np.clip(np.mean(np.vstack(probs), axis=0), 0.0, 1.0)

    def _mean_importance(models, cols):
        imps = [m.feature_importances_ for m in models]
        imp = np.mean(np.vstack(imps), axis=0)
        return pd.Series(imp, index=cols).sort_values(ascending=False)

    def _blend_probs(frame, tech_models, sent_models):
        p_tech = _predict_rf_ensemble(tech_models, frame, tech_cols)
        if sent_models is None:
            return p_tech

        p_sent = _predict_rf_ensemble(sent_models, frame, sent_cols)
        w_sent = np.full(len(frame), 0.35, dtype=float)
        finbert_col = f'FinBERT_Prob_Z20{suffix}' if f'FinBERT_Prob_Z20{suffix}' in frame.columns else ('FinBERT_Prob_Z20' if 'FinBERT_Prob_Z20' in frame.columns else None)
        if finbert_col is not None:
            z = pd.to_numeric(frame[finbert_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w_sent += 0.25 * np.clip(np.abs(z) / 2.0, 0.0, 1.0)
        adx_col = f'ADX_Z20{suffix}' if f'ADX_Z20{suffix}' in frame.columns else ('ADX_Z20' if 'ADX_Z20' in frame.columns else None)
        if adx_col is not None:
            adx_z = pd.to_numeric(frame[adx_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w_sent -= 0.12 * (adx_z < -0.5).astype(float)
        vix_col = f'VIX_Rel20{suffix}' if f'VIX_Rel20{suffix}' in frame.columns else ('VIX_Rel20' if 'VIX_Rel20' in frame.columns else None)
        if vix_col is not None:
            vix_rel = pd.to_numeric(frame[vix_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            w_sent += 0.10 * (vix_rel > 0.20).astype(float)
            w_sent -= 0.05 * (vix_rel < -0.10).astype(float)
        w_sent = np.clip(w_sent, 0.15, 0.75)
        return np.clip(w_sent * p_sent + (1.0 - w_sent) * p_tech, 0.0, 1.0)

    def _to_conviction(train_probs, probs):
        base = np.asarray(train_probs, dtype=float)
        base = np.sort(base[np.isfinite(base)])
        raw = np.asarray(probs, dtype=float)
        raw = np.clip(np.nan_to_num(raw, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if base.size < 20:
            return raw
        ranks = np.searchsorted(base, raw, side='right') / float(base.size)
        return np.clip(0.05 + 0.90 * ranks, 0.01, 0.99)

    def _fit_model_bundle(target_train_slice, seed_offset=0, start_date=None):
        local_target = target_train_slice.sort_index().copy()
        if start_date is not None:
            start_date = pd.Timestamp(start_date).tz_localize(None)
            local_target = local_target.loc[local_target.index >= start_date].copy()
        local_target = _sanitize_feature_frame(local_target)
        if anti_overfit_mode and purge_bars > 0 and len(local_target) > (min_train_rows + purge_bars):
            local_fit = local_target.iloc[:-purge_bars].copy()
        else:
            local_fit = local_target
        if len(local_fit) < min_train_rows:
            return None

        calib_size = max(40, min(80, int(len(local_fit) * 0.08)))
        if len(local_fit) > (min_train_rows + calib_size + 40):
            train_core = local_fit.iloc[:-calib_size].copy()
            calib_df = local_fit.iloc[-calib_size:].copy()
        else:
            train_core = local_fit.copy()
            calib_df = None

        cutoff = train_core.index.max()
        panel_train = _build_panel_train_frame(train_core, cutoff, start_date=start_date)
        if len(panel_train) < min_train_rows:
            return None
        y_train = panel_train['Target']
        if y_train.nunique() < 2:
            return None
        model_tech_local = _fit_rf_ensemble(panel_train, tech_cols, y_train, seed=42 + seed_offset)
        model_sent_local = None
        if sent_cols != tech_cols:
            model_sent_local = _fit_rf_ensemble(panel_train, sent_cols, y_train, seed=84 + seed_offset)

        flip_signal = False
        calib_auc = np.nan
        if calib_df is not None and len(calib_df) >= 80 and calib_df['Target'].nunique() >= 2:
            try:
                train_raw_core = _blend_probs(train_core, model_tech_local, model_sent_local)
                calib_raw = _blend_probs(calib_df, model_tech_local, model_sent_local)
                calib_prob = _to_conviction(train_raw_core, calib_raw)
                calib_auc = float(roc_auc_score(calib_df['Target'].to_numpy(dtype=int), calib_prob))
                flip_signal = calib_auc < 0.50
            except Exception:
                calib_auc = np.nan
                flip_signal = False
        return {
            'model_tech': model_tech_local,
            'model_sent': model_sent_local,
            'train_target': train_core,
            'flip_signal': flip_signal,
            'calib_auc': calib_auc,
        }

    backtest_data = None
    if train_window_mode == "single":
        train_cutoff = dev_fit_df.index.max()
        train_df = _build_panel_train_frame(dev_fit_df, train_cutoff, start_date=regime_start)
        if len(train_df) < min_train_rows:
            print("Error: training frame too small after sanitation.")
            return
        y_dev = train_df['Target']
        if y_dev.nunique() < 2:
            print("Error: training target has only one class.")
            return
        model_tech = _fit_rf_ensemble(train_df, tech_cols, y_dev, seed=42)
        model_sent = None
        if sent_cols != tech_cols:
            model_sent = _fit_rf_ensemble(train_df, sent_cols, y_dev, seed=84)

        tech_importance = _mean_importance(model_tech, tech_cols)
        print("\nTech/regime feature importance:")
        print(tech_importance.head(12))
        if model_sent is not None:
            sent_importance = _mean_importance(model_sent, sent_cols)
            print("\nSentiment feature importance:")
            print(sent_importance.head(10))

        dev_raw_probs = _blend_probs(dev_fit_df, model_tech, model_sent)
        test_raw_probs = _blend_probs(holdout_df, model_tech, model_sent)
        test_probs = _to_conviction(dev_raw_probs, test_raw_probs)
        backtest_data = holdout_df.copy()
        backtest_data['Pred_Prob'] = test_probs
        backtest_data['Meta_Prob'] = 1.0
    else:
        year_slices = sorted(pd.Index(holdout_df.index.year).unique().tolist())
        segment_frames = []
        tech_imp_slices = []
        sent_imp_slices = []
        for step_idx, y in enumerate(year_slices, start=1):
            test_seg = holdout_df.loc[holdout_df.index.year == y].copy()
            if test_seg.empty:
                continue
            seg_start = test_seg.index.min()
            train_slice = df.loc[df.index < seg_start].copy()
            if train_window_mode == "rolling":
                lookback_years = int(max(1, rolling_train_years))
                start_bound = seg_start - pd.DateOffset(years=lookback_years)
            else:
                start_bound = regime_start
            bundle = _fit_model_bundle(train_slice, seed_offset=step_idx, start_date=start_bound)
            if bundle is None:
                print(
                    f"Walk-forward {train_window_mode} step skipped: year={y}, "
                    "insufficient train rows after purge/filter."
                )
                continue
            model_tech = bundle['model_tech']
            model_sent = bundle['model_sent']
            train_target = bundle['train_target']
            flip_signal = bool(bundle.get('flip_signal', False))
            calib_auc = bundle.get('calib_auc', np.nan)
            tech_imp_slices.append(_mean_importance(model_tech, tech_cols))
            if model_sent is not None:
                sent_imp_slices.append(_mean_importance(model_sent, sent_cols))

            train_raw = _blend_probs(train_target, model_tech, model_sent)
            seg_raw = _blend_probs(test_seg, model_tech, model_sent)
            seg_probs = _to_conviction(train_raw, seg_raw)
            if flip_signal:
                seg_probs = 1.0 - seg_probs
            seg_out = test_seg.copy()
            seg_out['Pred_Prob'] = seg_probs
            seg_out['Meta_Prob'] = 1.0
            segment_frames.append(seg_out)
            print(
                f"Walk-forward {train_window_mode} step {step_idx}: "
                f"train {train_target.index.min().date()} -> {train_target.index.max().date()}, "
                f"test year={y}, rows={len(seg_out)}, "
                f"calib_auc={calib_auc:.3f}, flip={flip_signal}"
            )

        if not segment_frames:
            print(f"Error: no valid walk-forward prediction segments for mode={train_window_mode}.")
            return

        if tech_imp_slices:
            tech_importance = pd.concat(tech_imp_slices, axis=1).mean(axis=1).sort_values(ascending=False)
            print("\nTech/regime feature importance (walk-forward mean):")
            print(tech_importance.head(12))
        if sent_imp_slices:
            sent_importance = pd.concat(sent_imp_slices, axis=1).mean(axis=1).sort_values(ascending=False)
            print("\nSentiment feature importance (walk-forward mean):")
            print(sent_importance.head(10))

        backtest_data = pd.concat(segment_frames, axis=0).sort_index()

    # Walk-forward slices on development years for parameter selection only.
    wf_slices = []
    if anti_overfit_mode:
        dev_years = sorted(pd.Index(dev_fit_df.index.year).unique().tolist())
        wf_test_years = dev_years[-3:] if len(dev_years) >= 4 else dev_years[1:]
        for fold_idx, test_year in enumerate(wf_test_years, start=1):
            val_positions = np.flatnonzero(dev_fit_df.index.year == test_year)
            if val_positions.size < 80:
                continue
            val_start = int(val_positions[0])
            train_end = max(0, val_start - purge_bars)
            if train_end < 300:
                continue
            train_mask = np.zeros(len(dev_fit_df), dtype=bool)
            train_mask[:train_end] = True
            val_mask = np.zeros(len(dev_fit_df), dtype=bool)
            val_mask[val_positions] = True
            train_target_fold = dev_fit_df.loc[train_mask].copy()
            val_df_fold = _sanitize_feature_frame(dev_fit_df.loc[val_mask].copy())
            if train_target_fold.empty:
                continue
            fold_cutoff = train_target_fold.index.max()
            train_df_fold = _build_panel_train_frame(train_target_fold, fold_cutoff)
            y_train_fold = train_df_fold['Target']
            if y_train_fold.nunique() < 2:
                continue

            fold_tech_model = _fit_rf_ensemble(train_df_fold, tech_cols, y_train_fold, seed=42 + fold_idx)
            fold_sent_model = None
            if sent_cols != tech_cols:
                fold_sent_model = _fit_rf_ensemble(train_df_fold, sent_cols, y_train_fold, seed=84 + fold_idx)
            train_raw = _blend_probs(train_df_fold, fold_tech_model, fold_sent_model)
            val_raw = _blend_probs(val_df_fold, fold_tech_model, fold_sent_model)
            fold_probs = _to_conviction(train_raw, val_raw)
            fold_df = val_df_fold.copy()
            fold_df['Pred_Prob'] = fold_probs
            fold_df['Meta_Prob'] = 1.0
            wf_slices.append((test_year, fold_df))
            print(
                f"Walk-forward fold {fold_idx}: train years <= {test_year - 1} "
                f"(purged {purge_bars} bars), test year={test_year}, rows={len(fold_df)}"
            )
        if not wf_slices:
            print("Walk-forward warning: no valid dev folds built; optimization will use holdout directly.")

    print("\nPrediction probability summary:")
    print(backtest_data['Pred_Prob'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    # --- E. Run backtest ---
    print(f"\nBacktest window: {backtest_data.index.min().date()} -> {backtest_data.index.max().date()}")
    
    # Explicit next-bar execution and market friction assumptions.
    bt = Backtest(
        backtest_data,
        MLStrategy,
        cash=10000,
        commission=DEFAULT_COMMISSION,
        spread=DEFAULT_SPREAD,
        margin=1.0 / MLStrategy.max_gross_leverage,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    selected_params = {}
    if manual_params:
        print("\nManual parameter run enabled (skipping search).")
        selected_params = dict(manual_params)
        stats = bt.run(**selected_params)
        print("Manual params:")
        for k, v in selected_params.items():
            print(f"  - {k}: {v}")
    elif optimize_params:
        print("\nOptimizing strategy parameters...")
        btb._tqdm = lambda seq, *args, **kwargs: seq

        if anti_overfit_mode:
            # Strong pruning of parameter dimensionality to reduce multiple-testing bias.
            search_space = {
                'confidence_entry': [0.50, 0.52, 0.55, 0.58, 0.60],
                'entry_quantile': [0.25, 0.35, 0.45, 0.55],
                'macro_relaxed_entry': [0.50, 0.52, 0.55],
                'min_exposure': [0.08, 0.10, 0.15],
                'max_exposure': [1.00, 1.25, 1.50, 1.60, 1.80],
                'target_vol_annual': [0.18, 0.25, 0.35, 0.45],
                'rebalance_tolerance': [0.10, 0.12, 0.15, 0.20],
                'trail_atr_mult': [1.8, 2.2, 2.6, 3.0],
                'entry_sl_atr_mult': [0.9, 1.2, 1.4, 1.8],
                'min_trail_atr_mult': [1.0, 1.2, 1.4],
                'time_stop_bars': [10, 12, 15, 20, 30],
                'loss_cut_pct': [0.04, 0.06, 0.07, 0.08, 0.10],
                'max_position_drawdown': [0.08, 0.10, 0.12, 0.15],
                'vix_limit': [18.0, 20.0, 22.0, 24.0],
                'regime_require_both': [False, True],
                'trend_buffer': [0.97, 0.98],
                'trend_exit_buffer': [0.97, 0.98],
                'max_pyramids': [0, 1, 2, 3],
                'adx_entry_min': [6.0, 8.0, 10.0],
                'adx_chop_level': [20.0, 22.0, 25.0],
                'finbert_min_prob': [0.40, 0.42, 0.45, 0.48, 0.50],
                'finbert_z_min': [-1.50, -1.00, -0.50, -0.20],
                'drawdown_pause': [0.10, 0.12, 0.15],
                'cooldown_bars': [0, 1, 2, 3, 5],
                'chop_filter_enabled': [False, True],
                'chop_adx_limit': [16.0, 18.0, 20.0],
                'chop_band_atr': [0.35, 0.50, 0.65, 0.80],
                'chop_long_prob': [0.20, 0.30, 0.40, 0.50],
                'chop_short_prob': [0.50, 0.60, 0.70, 0.80],
                'chop_max_exposure': [0.20, 0.30, 0.40],
                'short_enabled': [False, True],
                'short_entry_quantile': [0.20, 0.25, 0.30],
                'short_min_size': [0.05, 0.08, 0.10],
                'short_max_size': [0.25, 0.35, 0.45]
            }
        else:
            search_space = {
                'confidence_entry': [0.20, 0.25, 0.30, 0.35],
                'entry_quantile': [0.65, 0.75, 0.85],
                'macro_relaxed_entry': [0.45, 0.50],
                'meta_threshold': [0.35, 0.45, 0.55],
                'finbert_min_prob': [0.25, 0.30, 0.35, 0.40],
                'finbert_z_min': [-1.00, -0.50, 0.00, 0.50],
                'adx_entry_min': [0.0, 5.0, 10.0, 15.0],
                'vix_limit': [24.0, 30.0, 35.0],
                'min_peer_rs': [-0.16, -0.08, 0.00],
                'putcall_max': [1.20, 1.40],
                'min_exposure': [0.10, 0.15],
                'max_exposure': [2.20, 2.60, 3.00],
                'target_vol_annual': [0.70, 0.90, 1.20],
                'rebalance_tolerance': [0.30, 0.50, 0.70],
                'trail_atr_mult': [3.0, 3.5, 4.0],
                'entry_sl_atr_mult': [1.0, 1.4, 1.8],
                'min_trail_atr_mult': [1.0, 1.2],
                'time_tighten_start': [20, 30],
                'time_tighten_step': [10, 20],
                'time_tighten_factor': [0.95, 0.98],
                'time_stop_bars': [45, 90, 120],
                'loss_cut_pct': [0.10, 0.12, 0.15],
                'max_position_drawdown': [0.20, 0.25, 0.30],
                'profit_lock_activate': [0.10],
                'max_profit_giveback': [0.30, 0.35],
                'drawdown_pause': [0.15, 0.20, 0.25],
                'cooldown_bars': [0, 1, 3],
                'regime_require_both': [False],
                'exit_on_regime_break': [False],
                'trend_buffer': [0.94, 0.96],
                'trend_exit_buffer': [0.94, 0.96, 0.98],
                'chop_filter_enabled': [False],
                'adx_chop_level': [20.0, 25.0, 30.0],
                'chop_stop_tighten': [0.85, 0.95],
                'prob_lookback': [63, 126],
                'pyramid_add_exposure': [0.20, 0.30],
                'max_pyramids': [2, 4, 6],
                'short_enabled': [False]
            }

        use_walkforward_opt = anti_overfit_mode and len(wf_slices) >= 2
        if use_walkforward_opt:
            print(
                f"Using walk-forward optimization on {len(wf_slices)} dev slices; "
                "final holdout remains untouched."
            )

        requested_samples = int(max(1, optimize_samples))
        if use_walkforward_opt:
            n_samples = min(requested_samples, 2000)
            if n_samples != requested_samples:
                print(
                    f"Anti-overfit guard: capping optimize_samples to {n_samples} "
                    f"(requested {requested_samples})."
                )
        else:
            if requested_samples != 5000:
                print(f"Overfit guard: forcing optimize_samples=5000 (requested {requested_samples}).")
            n_samples = 5000

        rng = np.random.default_rng(optimize_seed)

        def _pick(values):
            value = rng.choice(values)
            if isinstance(value, np.generic):
                return value.item()
            return value

        def _safe_mean(values, default=np.nan):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float(default)
            return float(arr.mean())

        best_score = -np.inf
        best_params = None
        best_metrics = None
        goal_best_score = -np.inf
        goal_best_params = None
        goal_best_metrics = None
        goal_hits = 0
        min_trade_target = 0
        constrained_best_score = -np.inf
        constrained_best_params = None
        constrained_best_metrics = None
        tested = 0
        progress_step = 100 if n_samples >= 400 else 50
        optimize_start = time.time()

        for _ in range(n_samples):
            params = {k: _pick(v) for k, v in search_space.items()}
            if params['min_exposure'] >= params['max_exposure']:
                continue
            if params['max_exposure'] > MLStrategy.max_gross_leverage:
                continue
            if params['min_trail_atr_mult'] > params['trail_atr_mult']:
                continue
            if (
                params.get('short_enabled', False)
                and params.get('short_min_size', 0.0) >= params.get('short_max_size', 1.0)
            ):
                continue

            tested += 1
            if use_walkforward_opt:
                fold_metrics = []
                fold_failed = False
                for _, fold_data in wf_slices:
                    bt_fold = Backtest(
                        fold_data,
                        MLStrategy,
                        cash=10000,
                        commission=DEFAULT_COMMISSION,
                        spread=DEFAULT_SPREAD,
                        margin=1.0 / MLStrategy.max_gross_leverage,
                        trade_on_close=False,
                        exclusive_orders=True,
                        finalize_trades=True,
                    )
                    try:
                        fold_stats = bt_fold.run(**params)
                    except Exception:
                        fold_failed = True
                        break
                    fold_metrics.append(fold_stats)

                if fold_failed or not fold_metrics:
                    continue

                sharpe = _safe_mean([m['Sharpe Ratio'] for m in fold_metrics], default=-10.0)
                sortino = _safe_mean([m['Sortino Ratio'] for m in fold_metrics], default=-10.0)
                calmar = _safe_mean([m['Calmar Ratio'] for m in fold_metrics], default=-10.0)
                ret = _safe_mean([m['Return [%]'] for m in fold_metrics], default=-200.0)
                bh = _safe_mean([m['Buy & Hold Return [%]'] for m in fold_metrics], default=1.0)
                max_dd = _safe_mean([abs(m['Max. Drawdown [%]']) for m in fold_metrics], default=100.0)
                trades = int(np.nansum([m['# Trades'] for m in fold_metrics]))
                exposure = _safe_mean([m['Exposure Time [%]'] for m in fold_metrics], default=0.0)
                ret_arr = np.asarray([m['Return [%]'] for m in fold_metrics], dtype=float)
                ret_arr = ret_arr[np.isfinite(ret_arr)]
                ret_std = float(ret_arr.std()) if ret_arr.size else 100.0
                sharpe_arr = np.asarray([m['Sharpe Ratio'] for m in fold_metrics], dtype=float)
                sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]
                ret_min = float(ret_arr.min()) if ret_arr.size else -200.0
                sharpe_min = float(sharpe_arr.min()) if sharpe_arr.size else -10.0
            else:
                try:
                    candidate = bt.run(**params)
                except Exception:
                    continue
                sharpe = float(candidate['Sharpe Ratio']) if pd.notna(candidate['Sharpe Ratio']) else -10.0
                sortino = float(candidate['Sortino Ratio']) if pd.notna(candidate['Sortino Ratio']) else -10.0
                calmar = float(candidate['Calmar Ratio']) if pd.notna(candidate['Calmar Ratio']) else -10.0
                ret = float(candidate['Return [%]']) if pd.notna(candidate['Return [%]']) else -200.0
                bh = float(candidate['Buy & Hold Return [%]']) if pd.notna(candidate['Buy & Hold Return [%]']) else 1.0
                max_dd = abs(float(candidate['Max. Drawdown [%]'])) if pd.notna(candidate['Max. Drawdown [%]']) else 100.0
                trades = int(candidate['# Trades']) if pd.notna(candidate['# Trades']) else 0
                exposure = float(candidate['Exposure Time [%]']) if pd.notna(candidate['Exposure Time [%]']) else 0.0
                ret_std = 0.0
                ret_min = ret
                sharpe_min = sharpe

            ret_ratio = ret / max(1.0, bh)
            ret_edge = ret_ratio - 1.0
            activity_bonus = 0.10 * min(trades, 80)
            activity_penalty = 0.0
            if trades < 20:
                activity_penalty += 10.0
            elif trades < 30:
                activity_penalty += 4.0
            elif trades > 140:
                activity_penalty += 0.05 * (trades - 140)
            score = (
                4.0 * sharpe
                + 1.0 * sortino
                + 0.5 * calmar
                + 20.0 * ret_edge
                + 0.0010 * ret
                - 2.0 * max(0.0, max_dd - 12.0)
                - 0.08 * ret_std
                + 0.03 * ret_min
                + 0.8 * sharpe_min
                + activity_bonus
                - activity_penalty
                - (1.0 if exposure < 8.0 else 0.0)
            )
            if ret <= bh:
                score -= 6.0
            if max_dd > 20.0:
                score -= 8.0
            if ret_min < -5.0:
                score -= 4.0
            if sharpe_min < -0.5:
                score -= 3.0

            meets_goal = (
                sharpe >= 1.0
                and ret > bh
                and max_dd <= 25.0
                and trades >= 15
            )
            if meets_goal:
                goal_hits += 1
                goal_score = (
                    6.0 * sharpe
                    + 0.10 * (ret - bh)
                    + 0.02 * ret
                    - 0.20 * max_dd
                    + 0.03 * min(trades, 80)
                )
                if goal_score > goal_best_score:
                    goal_best_score = goal_score
                    goal_best_params = params.copy()
                    goal_best_metrics = {
                        'Sharpe': sharpe,
                        'Return': ret,
                        'BuyHold': bh,
                        'MaxDD': max_dd,
                        'Trades': trades,
                    }

            meets_constraints = (
                trades >= min_trade_target
                and ret > 0.0
                and max_dd <= 20.0
            )
            if meets_constraints and score > constrained_best_score:
                constrained_best_score = score
                constrained_best_params = params.copy()
                constrained_best_metrics = {
                    'Sharpe': sharpe,
                    'Return': ret,
                    'BuyHold': bh,
                    'MaxDD': max_dd,
                    'Trades': trades,
                }

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_metrics = {
                    'Sharpe': sharpe,
                    'Return': ret,
                    'BuyHold': bh,
                    'MaxDD': max_dd,
                    'Trades': trades,
                }

            if tested % progress_step == 0:
                elapsed = time.time() - optimize_start
                if best_metrics is None:
                    print(f"  Optimization progress: tested {tested}/{n_samples} ({elapsed:.1f}s), best pending...")
                else:
                    print(
                        "  Optimization progress: "
                        f"tested {tested}/{n_samples} ({elapsed:.1f}s), "
                        f"best Sharpe={best_metrics['Sharpe']:.4f}, "
                        f"Return={best_metrics['Return']:.2f}%, "
                        f"MaxDD=-{best_metrics['MaxDD']:.2f}%, "
                        f"Trades={best_metrics['Trades']}"
                    )
                if goal_best_metrics is not None:
                    print(
                        "    Goal hits: "
                        f"{goal_hits}, best-goal Sharpe={goal_best_metrics['Sharpe']:.4f}, "
                        f"Return={goal_best_metrics['Return']:.2f}% "
                        f"vs Buy&Hold={goal_best_metrics['BuyHold']:.2f}%"
                    )
                elif goal_hits > 0:
                    print(f"    Goal hits: {goal_hits}, waiting for stable best-goal candidate...")

        use_constrained = False
        goal_selected = False
        if goal_best_params is not None and goal_best_metrics is not None:
            best_params = goal_best_params
            best_metrics = goal_best_metrics
            best_score = goal_best_score
            goal_selected = True
            print(
                "Goal-target mode: selected candidate meeting "
                "Sharpe>1 and Return>Buy&Hold."
            )
        elif goal_hits > 0:
            print(
                "Goal-target mode: qualifying candidates existed but none were stable enough; "
                "falling back to utility-based selection."
            )
        if (not goal_selected) and constrained_best_params is not None and constrained_best_metrics is not None:
            if best_metrics is None:
                use_constrained = True
            else:
                best_utility = (
                    3.0 * best_metrics['Sharpe']
                    + 0.03 * best_metrics['Return']
                    - 0.20 * best_metrics['MaxDD']
                    + 0.04 * min(best_metrics['Trades'], 40)
                )
                constrained_utility = (
                    3.0 * constrained_best_metrics['Sharpe']
                    + 0.03 * constrained_best_metrics['Return']
                    - 0.20 * constrained_best_metrics['MaxDD']
                    + 0.04 * min(constrained_best_metrics['Trades'], 40)
                )
                use_constrained = constrained_utility >= best_utility
            if use_constrained:
                best_params = constrained_best_params
                best_metrics = constrained_best_metrics
                best_score = constrained_best_score

        if best_params is None:
            print("No valid sampled parameter set found. Falling back to baseline run.")
            stats = bt.run()
        else:
            selected_params = best_params.copy()
            stats = bt.run(**selected_params)
            print(f"Optimization tested {tested}/{n_samples} sampled sets.")
            if constrained_best_params is not None and use_constrained:
                print(
                    f"Trade-target mode: selected candidate meeting min trades >= {min_trade_target} "
                    "on development folds."
                )
            elif constrained_best_params is not None and not use_constrained:
                print(
                    "Trade-target mode: constrained candidate found, "
                    "but unconstrained candidate had stronger risk-adjusted utility."
                )
            if use_walkforward_opt:
                print(
                    "Parameters selected on walk-forward development slices; "
                    "metrics below are untouched holdout results."
                )
            print("Selected params:")
            for k, v in selected_params.items():
                print(f"  - {k}: {v}")
            if best_metrics is not None:
                print(
                    "Selection metrics (dev): "
                    f"Sharpe={best_metrics['Sharpe']:.4f}, "
                    f"Return={best_metrics['Return']:.2f}%, "
                    f"Buy&Hold={best_metrics['BuyHold']:.2f}%, "
                    f"MaxDD=-{best_metrics['MaxDD']:.2f}%, "
                    f"Trades={best_metrics['Trades']}"
                )
            print(f"Composite score: {best_score:.4f}")
    else:
        print("\nOptimization disabled: running calibrated baseline parameter set.")
        stats = bt.run()

    # Persist selection metadata so external orchestrators can save ticker-specific params.
    if isinstance(selected_params, dict) and selected_params:
        stats['_selected_params'] = dict(selected_params)
    else:
        stats['_selected_params'] = {}
    stats['_train_window_mode'] = str(train_window_mode)
    stats['_rolling_train_years'] = int(rolling_train_years)

    if anti_overfit_mode and noise_test and len(backtest_data) > 80:
        noisy_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'ATR_20', 'SMA_200', 'ADX_14'] if c in backtest_data.columns]
        if noisy_cols:
            param_kwargs = selected_params.copy() if selected_params else {}
            noise_records = []
            for run_idx in range(max(1, int(noise_runs))):
                noisy_df = backtest_data.copy()
                rng_noise = np.random.default_rng(8000 + run_idx)
                for col in noisy_cols:
                    vals = pd.to_numeric(noisy_df[col], errors='coerce').ffill().bfill()
                    eps = rng_noise.normal(0.0, float(noise_level), size=len(vals))
                    noisy_df[col] = vals.to_numpy(dtype=float) * (1.0 + eps)

                if {'Open', 'High', 'Low', 'Close'}.issubset(noisy_df.columns):
                    noisy_df['High'] = np.maximum.reduce([
                        noisy_df['High'].to_numpy(dtype=float),
                        noisy_df['Open'].to_numpy(dtype=float),
                        noisy_df['Close'].to_numpy(dtype=float),
                    ])
                    noisy_df['Low'] = np.minimum.reduce([
                        noisy_df['Low'].to_numpy(dtype=float),
                        noisy_df['Open'].to_numpy(dtype=float),
                        noisy_df['Close'].to_numpy(dtype=float),
                    ])
                    
                bt_noise = Backtest(
                    noisy_df,
                    MLStrategy,
                    cash=10000,
                    commission=DEFAULT_COMMISSION,
                    spread=DEFAULT_SPREAD,
                    margin=1.0 / MLStrategy.max_gross_leverage,
                    trade_on_close=False,
                    exclusive_orders=True,
                    finalize_trades=True,
                )
                noise_stats = bt_noise.run(**param_kwargs)
                noise_records.append({
                    'run': run_idx + 1,
                    'return_pct': float(noise_stats['Return [%]']),
                    'sharpe': float(noise_stats['Sharpe Ratio']),
                    'max_dd_pct': float(noise_stats['Max. Drawdown [%]']),
                })

            noise_df = pd.DataFrame(noise_records)
            noise_path = PROCESSED_DIR / f"{ticker}_noise_injection_report.csv"
            noise_df.to_csv(noise_path, index=False)
            print(
                f"\nNoise injection ({noise_level:.4%}, runs={len(noise_df)}): "
                f"median Return={noise_df['return_pct'].median():.2f}%, "
                f"median Sharpe={noise_df['sharpe'].median():.4f}, "
                f"median MaxDD={noise_df['max_dd_pct'].median():.2f}%"
            )
            print(f"Noise report saved to: {noise_path}")

    
    bh_max_dd_pct = np.nan
    if 'Close' in backtest_data.columns:
        close_vals = pd.to_numeric(backtest_data['Close'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        close_vals = close_vals[close_vals > 0]
        if not close_vals.empty:
            bh_curve = close_vals / float(close_vals.iloc[0])
            bh_dd = (bh_curve / bh_curve.cummax()) - 1.0
            bh_max_dd_pct = float(bh_dd.min() * 100.0)
    stats['Buy & Hold Max. Drawdown [%]'] = bh_max_dd_pct

    def _equity_curve_metrics(equity_series):
        eq = pd.to_numeric(equity_series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if eq.empty:
            return np.nan, np.nan, np.nan
        ret_pct = float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0)
        daily = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily) > 1 and daily.std(ddof=0) > 0:
            sharpe = float(np.sqrt(252.0) * daily.mean() / daily.std(ddof=0))
        else:
            sharpe = np.nan
        dd = (eq / eq.cummax() - 1.0) * 100.0
        max_dd_pct = float(dd.min()) if not dd.empty else np.nan
        return ret_pct, sharpe, max_dd_pct

    can_attach_combo = (
        bool(attach_combo_summary)
        and manual_params is None
        and not optimize_params
    )
    if can_attach_combo:
        balanced_path = PROCESSED_DIR / f"{ticker}_three_window_best_params.json"
        aggressive_path = PROCESSED_DIR / f"{ticker}_three_window_best_params_v2.json"
        if balanced_path.exists() and aggressive_path.exists():
            try:
                balanced_cfg = json.loads(balanced_path.read_text())
                aggressive_cfg = json.loads(aggressive_path.read_text())
                balanced_params = dict(balanced_cfg.get('params', {}))
                aggressive_params = dict(aggressive_cfg.get('params', {}))
                balanced_years = int(balanced_cfg.get('rolling_train_years', 4))
                aggressive_years = int(aggressive_cfg.get('rolling_train_years', 3))

                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        bal_stats = backtester1(
                            ticker=ticker,
                            optimize_params=False,
                            optimize_samples=optimize_samples,
                            optimize_seed=optimize_seed,
                            anti_overfit_mode=anti_overfit_mode,
                            noise_test=False,
                            noise_level=noise_level,
                            noise_runs=noise_runs,
                            purge_bars=purge_bars,
                            manual_params=balanced_params,
                            enable_panel_training=enable_panel_training,
                            panel_tickers=panel_tickers,
                            eval_start=eval_start,
                            eval_end=eval_end,
                            train_window_mode="rolling",
                            rolling_train_years=balanced_years,
                            train_start=train_start,
                            save_report=False,
                            print_summary=False,
                            attach_combo_summary=False,
                            combo_weight_balanced=combo_weight_balanced,
                        )
                        agg_stats = backtester1(
                            ticker=ticker,
                            optimize_params=False,
                            optimize_samples=optimize_samples,
                            optimize_seed=optimize_seed,
                            anti_overfit_mode=anti_overfit_mode,
                            noise_test=False,
                            noise_level=noise_level,
                            noise_runs=noise_runs,
                            purge_bars=purge_bars,
                            manual_params=aggressive_params,
                            enable_panel_training=enable_panel_training,
                            panel_tickers=panel_tickers,
                            eval_start=eval_start,
                            eval_end=eval_end,
                            train_window_mode="rolling",
                            rolling_train_years=aggressive_years,
                            train_start=train_start,
                            save_report=False,
                            print_summary=False,
                            attach_combo_summary=False,
                            combo_weight_balanced=combo_weight_balanced,
                        )

                stats['Balanced Return [%]'] = float(bal_stats.get('Return [%]', np.nan))
                stats['Balanced Sharpe Ratio'] = float(bal_stats.get('Sharpe Ratio', np.nan))
                stats['Balanced Max. Drawdown [%]'] = float(bal_stats.get('Max. Drawdown [%]', np.nan))
                stats['Aggressive Return [%]'] = float(agg_stats.get('Return [%]', np.nan))
                stats['Aggressive Sharpe Ratio'] = float(agg_stats.get('Sharpe Ratio', np.nan))
                stats['Aggressive Max. Drawdown [%]'] = float(agg_stats.get('Max. Drawdown [%]', np.nan))

                bal_eq = bal_stats.get('_equity_curve', pd.DataFrame()).get('Equity')
                agg_eq = agg_stats.get('_equity_curve', pd.DataFrame()).get('Equity')
                if bal_eq is not None and agg_eq is not None:
                    combo_w = float(np.clip(combo_weight_balanced, 0.0, 1.0))
                    combo_curve = pd.concat(
                        [
                            pd.to_numeric(bal_eq, errors='coerce').rename('bal'),
                            pd.to_numeric(agg_eq, errors='coerce').rename('agg'),
                        ],
                        axis=1
                    ).sort_index().ffill().bfill()
                    combo_curve['combo'] = combo_w * combo_curve['bal'] + (1.0 - combo_w) * combo_curve['agg']
                    combo_ret, combo_sharpe, combo_dd = _equity_curve_metrics(combo_curve['combo'])
                    stats['Combo 50/50 Return [%]'] = combo_ret
                    stats['Combo 50/50 Sharpe Ratio'] = combo_sharpe
                    stats['Combo 50/50 Max. Drawdown [%]'] = combo_dd
            except Exception as combo_err:
                if print_summary:
                    print(f"\nCombo summary injection skipped: {combo_err}")
        else:
            if print_summary:
                print(
                    "\nCombo summary injection skipped: missing params files "
                    f"({balanced_path.name}, {aggressive_path.name})."
                )

    if print_summary:
        print("\nBacktest summary")
        print("-" * 32)
        print(stats)

    if save_report:
        save_path = PROCESSED_DIR / f"{ticker}_ML_Final_Fixed.html"
        bt.plot(filename=str(save_path), open_browser=False)
        if print_summary:
            print(f"\nReport saved to: {save_path}")
    if return_backtest_data:
        stats['_backtest_data'] = backtest_data.copy()
    return stats

# ==========================================
# 4. Standalone execution
# ==========================================
if __name__ == "__main__":
    backtester1("META")
