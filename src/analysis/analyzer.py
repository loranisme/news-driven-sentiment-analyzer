import pandas as pd
import numpy as np
from textblob import TextBlob
import os
import sys
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR, PROCESSED_DIR, MODEL_DIR

# ==========================================
# 2. Core analyzer
# ==========================================
class AdaptiveSentimentAnalyzer:
    def __init__(self, ticker="AAPL", regime_start="2014-01-01"):
        self.ticker = ticker
        parsed_regime = pd.to_datetime(regime_start, errors='coerce')
        if pd.isna(parsed_regime):
            parsed_regime = pd.Timestamp("2014-01-01")
        self.regime_start = pd.Timestamp(parsed_regime).date()
        
        # Dynamic input paths
        self.news_path = RAW_DIR / f"{ticker}_news.csv"
        self.stock_path = RAW_DIR / f"{ticker}_history.csv"
        
        # Dynamic output paths
        self.output_sentiment_path = PROCESSED_DIR / f"{ticker}_sentiment_daily.csv"
        self.final_merged_path = PROCESSED_DIR / f"{ticker}_sentiment_stock_merged.csv"
        
        # Ensure the output directory exists.
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def _calculate_base_sentiment(self, text):
        if not isinstance(text, str):
            return 0
        return TextBlob(text).sentiment.polarity

    def _compute_adx14(self, df):
        """
        Wilder-style ADX(14) using OHLC columns when available.
        Returns NaN series when high/low inputs are missing.
        """
        high_col = 'High' if 'High' in df.columns else ('high' if 'high' in df.columns else None)
        low_col = 'Low' if 'Low' in df.columns else ('low' if 'low' in df.columns else None)
        if high_col is None or low_col is None or 'Close' not in df.columns:
            return pd.Series(np.nan, index=df.index)

        high = pd.to_numeric(df[high_col], errors='coerce')
        low = pd.to_numeric(df[low_col], errors='coerce')
        close = pd.to_numeric(df['Close'], errors='coerce')
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1
        ).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index
        )

        atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx14 = dx.ewm(alpha=1 / 14, adjust=False).mean()
        return adx14

    def _fractional_diff(self, series, d=0.4, thresh=1e-4):
        """
        Fractional differentiation:
        keep long memory while improving stationarity.
        """
        s = pd.Series(series, dtype=float)
        weights = [1.0]
        k = 1
        while k < 2048:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            weights.append(w_k)
            k += 1

        w = np.array(weights[::-1], dtype=float)
        width = len(w)
        out = np.full(len(s), np.nan, dtype=float)
        vals = s.to_numpy(dtype=float)

        for i in range(width - 1, len(vals)):
            window = vals[i - width + 1:i + 1]
            if np.any(~np.isfinite(window)):
                continue
            out[i] = float(np.dot(w, window))

        return pd.Series(out, index=s.index)

    def _load_news_sentiment_series(self, symbol, out_col, keywords=None):
        """
        Build daily sentiment from a symbol-specific news file when available.
        """
        path = RAW_DIR / f"{symbol}_news.csv"
        if not os.path.exists(path):
            return None

        ndf = pd.read_csv(path)
        ndf.columns = [c.lower().strip() for c in ndf.columns]
        if 'date' not in ndf.columns or 'title' not in ndf.columns:
            return None

        ndf['date'] = pd.to_datetime(ndf['date'], errors='coerce', utc=True).dt.date
        ndf = ndf.dropna(subset=['date', 'title']).copy()
        if ndf.empty:
            return None

        if keywords:
            pat = '|'.join([k.replace('.', r'\.') for k in keywords])
            ndf = ndf[ndf['title'].astype(str).str.contains(pat, case=False, regex=True)]
            if ndf.empty:
                return None

        ndf['tmp_sent'] = ndf['title'].astype(str).apply(self._calculate_base_sentiment)
        daily = ndf.groupby('date', as_index=False)['tmp_sent'].mean()
        daily.rename(columns={'tmp_sent': out_col}, inplace=True)
        return daily

    def _load_external_close_series(self, symbol):
        """
        Load external market series (e.g., SPY/GOOGL/VIX) from raw history files.
        Returns a DataFrame with columns: date, close_ext.
        """
        candidates = [
            RAW_DIR / f"{symbol}_history.csv",
            RAW_DIR / f"{symbol.replace('^', '')}_history.csv",
        ]

        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            return None

        ext = pd.read_csv(path)
        ext.columns = [c.lower().strip() for c in ext.columns]
        if 'date' not in ext.columns or 'close' not in ext.columns:
            return None

        ext['date'] = pd.to_datetime(ext['date'], utc=True).dt.date
        ext = ext[['date', 'close']].copy()
        ext.rename(columns={'close': 'close_ext'}, inplace=True)
        return ext

    def _load_put_call_ratio_series(self, kind='total'):
        """
        Load CBOE put/call ratio series from raw files produced by price_collector.
        """
        path = RAW_DIR / f"CBOE_{kind}pc.csv"
        if not os.path.exists(path):
            return None

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'date' not in df.columns or 'put_call_ratio' not in df.columns:
            return None

        out = df[['date', 'put_call_ratio']].copy()
        out['date'] = pd.to_datetime(out['date'], errors='coerce', utc=True).dt.date
        out['put_call_ratio'] = pd.to_numeric(out['put_call_ratio'], errors='coerce')
        out = out.dropna(subset=['date', 'put_call_ratio']).copy()
        out = out.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        return out

    def _load_finbert_daily_prob(self):
        """
        Build daily FinBERT positive probabilities from ticker headlines.
        Uses local fine-tuned model when available; results cached to processed dir.
        """
        cache_path = PROCESSED_DIR / f"{self.ticker}_finbert_daily.csv"
        if not os.path.exists(self.news_path):
            return None

        ndf = pd.read_csv(self.news_path)
        ndf.columns = [c.lower().strip() for c in ndf.columns]
        if 'date' not in ndf.columns or 'title' not in ndf.columns:
            return None

        ndf['date'] = pd.to_datetime(ndf['date'], errors='coerce', utc=True).dt.date
        ndf['title'] = ndf['title'].astype(str).str.strip()
        ndf = ndf.dropna(subset=['date', 'title']).copy()
        ndf = ndf[ndf['title'].str.len() > 0].copy()
        if ndf.empty:
            return None

        # Reuse cache only when it fully covers the current news date span.
        if os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                cached.columns = [c.strip() for c in cached.columns]
                if 'date' in cached.columns and 'FinBERT_Prob' in cached.columns:
                    cached['date'] = pd.to_datetime(cached['date'], errors='coerce').dt.date
                    cached['FinBERT_Prob'] = pd.to_numeric(cached['FinBERT_Prob'], errors='coerce')
                    cached = cached.dropna(subset=['date', 'FinBERT_Prob']).copy()
                    if not cached.empty:
                        cache_min, cache_max = cached['date'].min(), cached['date'].max()
                        news_min, news_max = ndf['date'].min(), ndf['date'].max()
                        cache_span_ok = (cache_min <= news_min) and (cache_max >= news_max)
                        cache_density_ok = cached['date'].nunique() >= int(ndf['date'].nunique() * 0.80)
                        if cache_span_ok and cache_density_ok:
                            return cached[['date', 'FinBERT_Prob']]
                        print(
                            "[Analyzer] FinBERT cache stale/outdated for current news range; "
                            "recomputing daily probabilities."
                        )
            except Exception:
                pass

        model_dir = MODEL_DIR / f"{self.ticker}_finbert"
        if not model_dir.exists():
            return None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except Exception:
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
        except Exception:
            return None

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        model = model.to(device)
        model.eval()

        label2id = {str(k).upper(): int(v) for k, v in getattr(model.config, "label2id", {}).items()}
        up_idx = label2id.get("UP", 1 if int(getattr(model.config, "num_labels", 2)) >= 2 else 0)

        texts = ndf['title'].tolist()
        probs = np.full(len(texts), np.nan, dtype=float)
        bs = 64
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = texts[start:start + bs]
                tokens = tokenizer(
                    batch,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt"
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                logits = model(**tokens).logits
                p = torch.softmax(logits, dim=-1)[:, up_idx].detach().cpu().numpy()
                probs[start:start + len(batch)] = p

        ndf['FinBERT_Prob'] = probs
        daily = ndf.groupby('date', as_index=False)['FinBERT_Prob'].mean()
        daily = daily.dropna(subset=['date', 'FinBERT_Prob']).copy()
        if daily.empty:
            return None

        try:
            save_df = daily.copy()
            save_df['date'] = pd.to_datetime(save_df['date']).dt.strftime('%Y-%m-%d')
            save_df.to_csv(cache_path, index=False)
        except Exception:
            pass

        return daily

    def _add_technical_indicators(self, df):
        """
        Compute technical indicators such as moving averages, RSI, and momentum.
        """
        # Ensure the calculations use the Close column.
        if 'Close' not in df.columns:
            # Promote lowercase close to Close when needed.
            if 'close' in df.columns:
                df['Close'] = df['close']
            else:
                print("Warning: Close column not found. Technical indicators may be incomplete.")
                return df

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Momentum_5D'] = df['Close'].pct_change(periods=5)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill initial gaps from rolling calculations.
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def _add_orthogonal_features(self, df):
        """
        Add lower-correlation features from external market context when available.
        Missing external files are handled gracefully.
        """
        out = df.copy()
        out['Ret_1D'] = out['Close'].pct_change(fill_method=None)
        out['Ret_20D'] = out['Close'].pct_change(periods=20, fill_method=None)

        # 1) Peer relative strength (example peer: GOOGL)
        peer = self._load_external_close_series("GOOGL")
        if peer is not None:
            peer.rename(columns={'close_ext': 'GOOGL_Close'}, inplace=True)
            out = out.merge(peer, on='date', how='left')
            out['GOOGL_Close'] = out['GOOGL_Close'].ffill()
            out['GOOGL_Ret_20D'] = out['GOOGL_Close'].pct_change(periods=20, fill_method=None)
            out['Peer_RS_20D'] = out['Ret_20D'] - out['GOOGL_Ret_20D']
        else:
            out['Peer_RS_20D'] = 0.0

        # 1b) Sector/index relative strength: QQQ and SOXX
        for symbol in ['QQQ', 'SOXX']:
            bench = self._load_external_close_series(symbol)
            rs_col = f'RS_vs_{symbol}_20D'
            if bench is not None:
                close_col = f'{symbol}_Close'
                ret_col = f'{symbol}_Ret_20D'
                bench.rename(columns={'close_ext': close_col}, inplace=True)
                out = out.merge(bench, on='date', how='left')
                out[close_col] = out[close_col].ffill()
                out[ret_col] = out[close_col].pct_change(periods=20, fill_method=None)
                out[rs_col] = out['Ret_20D'] - out[ret_col]
                if symbol == 'QQQ':
                    out['QQQ_SMA_50'] = out[close_col].rolling(50).mean()
                    out['QQQ_Above_50DMA'] = (out[close_col] > out['QQQ_SMA_50']).astype(float)
            else:
                out[rs_col] = 0.0
                if symbol == 'QQQ':
                    out['QQQ_Above_50DMA'] = 1.0

        # 2) SPY regime: SPY above/below 200D moving average
        spy = self._load_external_close_series("SPY")
        if spy is not None:
            spy.rename(columns={'close_ext': 'SPY_Close'}, inplace=True)
            out = out.merge(spy, on='date', how='left')
            out['SPY_Close'] = out['SPY_Close'].ffill()
            out['SPY_SMA_200'] = out['SPY_Close'].rolling(200).mean()
            out['SPY_Above_200DMA'] = (out['SPY_Close'] > out['SPY_SMA_200']).astype(float)
        else:
            out['SPY_Above_200DMA'] = 1.0

        # 3) VIX stress filter
        vix = self._load_external_close_series("^VIX")
        if vix is None:
            vix = self._load_external_close_series("VIX")
        if vix is not None:
            vix.rename(columns={'close_ext': 'VIX_Close'}, inplace=True)
            out = out.merge(vix, on='date', how='left')
            out['VIX_Close'] = out['VIX_Close'].ffill()
            out['VIX_5D_Chg'] = out['VIX_Close'].pct_change(periods=5, fill_method=None)
            vix_roll_mean = out['VIX_Close'].rolling(20).mean()
            vix_roll_std = out['VIX_Close'].rolling(20).std()
            out['VIX_Z20'] = (out['VIX_Close'] - vix_roll_mean) / vix_roll_std
            out['VIX_Below_20'] = (out['VIX_Close'] < 20).astype(float)
        else:
            out['VIX_Close'] = 15.0
            out['VIX_5D_Chg'] = 0.0
            out['VIX_Z20'] = 0.0
            out['VIX_Below_20'] = 1.0

        # 3b) CBOE put/call ratios (market positioning / fear-greed overlay)
        pc_map = [('total', 'PutCall_Total'), ('equity', 'PutCall_Equity'), ('index', 'PutCall_Index')]
        for kind, col in pc_map:
            pc = self._load_put_call_ratio_series(kind=kind)
            if pc is not None:
                pc.rename(columns={'put_call_ratio': col}, inplace=True)
                out = out.merge(pc, on='date', how='left')
                out[col] = out[col].ffill()
            else:
                out[col] = np.nan

        if {'PutCall_Equity', 'PutCall_Index'}.issubset(out.columns):
            out['PutCall_EquityMinusIndex'] = out['PutCall_Equity'] - out['PutCall_Index']
        else:
            out['PutCall_EquityMinusIndex'] = 0.0

        if 'PutCall_Total' in out.columns:
            pc_mean = out['PutCall_Total'].rolling(20).mean()
            pc_std = out['PutCall_Total'].rolling(20).std()
            out['PutCall_Z20'] = (out['PutCall_Total'] - pc_mean) / pc_std
            out['PutCall_5D_Chg'] = out['PutCall_Total'].pct_change(periods=5, fill_method=None)
        else:
            out['PutCall_Total'] = 0.8
            out['PutCall_Z20'] = 0.0
            out['PutCall_5D_Chg'] = 0.0

        # 4) Volatility feature
        out['HV_20_Ann'] = out['Ret_1D'].rolling(20).std() * np.sqrt(252)

        # 4b) Fractional differentiation features (high-order stationarity with memory retention)
        out['FracDiff_Close_d04'] = self._fractional_diff(out['Close'], d=0.4)
        out['FracDiff_Close_d07'] = self._fractional_diff(out['Close'], d=0.7)
        if 'VIX_Close' in out.columns:
            out['FracDiff_VIX_d04'] = self._fractional_diff(out['VIX_Close'], d=0.4)
        else:
            out['FracDiff_VIX_d04'] = 0.0

        # 5) Order-flow proxy features from OHLCV
        if {'High', 'Low', 'Close', 'Volume'}.issubset(out.columns):
            hl_range = (out['High'] - out['Low']).replace(0, np.nan)
            mfm = ((out['Close'] - out['Low']) - (out['High'] - out['Close'])) / hl_range
            mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            mfv = mfm * out['Volume'].fillna(0.0)
            vol_sum = out['Volume'].rolling(20).sum().replace(0, np.nan)
            out['CMF_20'] = mfv.rolling(20).sum() / vol_sum

            obv = (np.sign(out['Ret_1D'].fillna(0.0)) * out['Volume'].fillna(0.0)).cumsum()
            out['OBV'] = obv
            out['OBV_Z20'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()

            log_dollar_vol = np.log1p((out['Close'].abs() * out['Volume'].fillna(0.0)).clip(lower=0.0))
            out['DollarVol_Z20'] = (log_dollar_vol - log_dollar_vol.rolling(20).mean()) / log_dollar_vol.rolling(20).std()
        else:
            out['CMF_20'] = 0.0
            out['OBV_Z20'] = 0.0
            out['DollarVol_Z20'] = 0.0

        # Risk-on filter: allow long only if VIX<20 OR SPY above 200DMA.
        out['Risk_On'] = ((out['VIX_Below_20'] > 0) | (out['SPY_Above_200DMA'] > 0)).astype(int)

        # 5b) FinBERT daily conviction (if local fine-tuned model exists)
        finbert_daily = self._load_finbert_daily_prob()
        if finbert_daily is not None:
            out = out.merge(finbert_daily, on='date', how='left')
            out['FinBERT_Prob'] = out['FinBERT_Prob'].ffill()
        else:
            out['FinBERT_Prob'] = np.nan
        sentiment_fallback = np.clip((out['sentiment_score'].fillna(0.0) + 1.0) / 2.0, 0.0, 1.0)
        out['FinBERT_Prob'] = out['FinBERT_Prob'].fillna(sentiment_fallback)

        # 6) Supply-chain and capex sentiment overlays (if corresponding news files exist)
        sc_specs = [
            ('TSM', 'SC_TSM_Sent', None),
            ('ASML', 'SC_ASML_Sent', None),
            ('SMCI', 'SC_SMCI_Sent', None),
            ('MSFT', 'SC_MSFT_CAPEX_Sent', ['capex', 'capital expenditure', 'ai investment', 'data center']),
            ('GOOGL', 'SC_GOOGL_CAPEX_Sent', ['capex', 'capital expenditure', 'ai investment', 'data center']),
            ('AMZN', 'SC_AMZN_CAPEX_Sent', ['capex', 'capital expenditure', 'ai investment', 'data center']),
        ]
        sc_cols = []
        for sym, col, keys in sc_specs:
            sc_df = self._load_news_sentiment_series(sym, col, keywords=keys)
            if sc_df is not None:
                out = out.merge(sc_df, on='date', how='left')
                out[col] = out[col].ffill()
            else:
                out[col] = 0.0
            sc_cols.append(col)

        if sc_cols:
            out['SupplyChain_Sent'] = out[sc_cols].mean(axis=1)
        else:
            out['SupplyChain_Sent'] = 0.0

        # Clean temporary helper columns.
        for col in [
            'GOOGL_Close', 'GOOGL_Ret_20D',
            'QQQ_Close', 'QQQ_Ret_20D',
            'SOXX_Close', 'SOXX_Ret_20D',
            'SPY_Close', 'SPY_SMA_200',
            'QQQ_SMA_50',
            'OBV'
        ]:
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        finbert_roll_mean = out['FinBERT_Prob'].rolling(20).mean()
        finbert_roll_std = out['FinBERT_Prob'].rolling(20).std().replace(0, np.nan)
        out['FinBERT_Prob_Z20'] = ((out['FinBERT_Prob'] - finbert_roll_mean) / finbert_roll_std).clip(-6, 6)
        out['FinBERT_Prob_Z20'] = out['FinBERT_Prob_Z20'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['FinBERT_Prob_5D_Chg'] = out['FinBERT_Prob'].pct_change(periods=5, fill_method=None)
        out['FinBERT_Prob_5D_Chg'] = out['FinBERT_Prob_5D_Chg'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _build_triple_barrier_target(
        self,
        df,
        atr_window=20,
        tp_atr_mult=1.8,
        sl_atr_mult=1.4,
        max_holding_bars=12,
        neutral_atr_mult=0.30,
    ):
        """
        Build binary labels with triple-barrier logic.
        Target = 1 only when take-profit is touched before stop-loss
        within the vertical barrier window; otherwise Target = 0.
        """
        if 'Close' not in df.columns:
            raise ValueError("Close column is required for triple-barrier target.")

        high_col = 'High' if 'High' in df.columns else ('high' if 'high' in df.columns else None)
        low_col = 'Low' if 'Low' in df.columns else ('low' if 'low' in df.columns else None)

        if high_col is None or low_col is None:
            print("Warning: High/Low columns not found. Falling back to Close-only barrier checks.")

        prev_close = df['Close'].shift(1)
        tr_parts = [
            ((df[high_col] if high_col else df['Close']) - (df[low_col] if low_col else df['Close'])).abs(),
            ((df[high_col] if high_col else df['Close']) - prev_close).abs(),
            ((df[low_col] if low_col else df['Close']) - prev_close).abs(),
        ]
        tr = pd.concat(tr_parts, axis=1).max(axis=1)
        atr = tr.ewm(span=atr_window, adjust=False).mean()
        adx14 = self._compute_adx14(df)
        vix_z = pd.to_numeric(df['VIX_Z20'], errors='coerce') if 'VIX_Z20' in df.columns else pd.Series(0.0, index=df.index)
        risk_on = pd.to_numeric(df['Risk_On'], errors='coerce') if 'Risk_On' in df.columns else pd.Series(1.0, index=df.index)

        close = df['Close'].to_numpy(dtype=float)
        high = (df[high_col].to_numpy(dtype=float) if high_col else close)
        low = (df[low_col].to_numpy(dtype=float) if low_col else close)
        atr_vals = atr.to_numpy(dtype=float)
        adx_vals = adx14.to_numpy(dtype=float)
        vix_vals = vix_z.to_numpy(dtype=float)
        risk_vals = risk_on.to_numpy(dtype=float)

        labels = np.full(len(df), np.nan, dtype=float)
        quality = np.full(len(df), np.nan, dtype=float)
        dynamic_tp = np.full(len(df), np.nan, dtype=float)
        dynamic_sl = np.full(len(df), np.nan, dtype=float)
        dynamic_hz = np.full(len(df), np.nan, dtype=float)
        vol_adj_move = np.full(len(df), np.nan, dtype=float)

        for i in range(len(df)):
            if i >= len(df) - 1:
                continue
            if not np.isfinite(atr_vals[i]) or atr_vals[i] <= 0:
                continue

            adx_i = adx_vals[i] if np.isfinite(adx_vals[i]) else 20.0
            vix_i = vix_vals[i] if np.isfinite(vix_vals[i]) else 0.0
            risk_i = risk_vals[i] if np.isfinite(risk_vals[i]) else 1.0

            trend_strength = float(np.clip((adx_i - 18.0) / 20.0, 0.0, 1.0))
            stress = float(np.clip((vix_i + 0.5) / 2.0, 0.0, 1.0))

            tp_mult = float(np.clip(tp_atr_mult * (1.0 + 0.55 * trend_strength - 0.20 * stress), 1.0, 3.5))
            sl_mult = float(np.clip(sl_atr_mult * (1.0 + 0.10 * stress - 0.15 * trend_strength), 0.8, 3.0))
            horizon = int(np.clip(round(max_holding_bars * (1.0 + 0.8 * trend_strength - 0.25 * stress)), 5, 30))

            dynamic_tp[i] = tp_mult
            dynamic_sl[i] = sl_mult
            dynamic_hz[i] = horizon

            if risk_i <= 0 and trend_strength < 0.20 and stress > 0.70:
                continue

            entry = close[i]
            tp_price = entry + tp_mult * atr_vals[i]
            sl_price = entry - sl_mult * atr_vals[i]
            end_idx = min(i + horizon, len(df) - 1)
            exit_idx = end_idx
            label = np.nan
            winner = None

            for j in range(i + 1, end_idx + 1):
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price

                if hit_tp and not hit_sl:
                    label = 1.0
                    winner = "tp"
                    exit_idx = j
                    break
                if hit_sl and not hit_tp:
                    label = 0.0
                    winner = "sl"
                    exit_idx = j
                    break
                if hit_tp and hit_sl:
                    # Conservative tie-breaker: if both touched intraday, treat as non-win.
                    label = 0.0
                    winner = "tie"
                    exit_idx = j
                    break

            # If neither barrier is touched, use vertical-barrier return only when move is meaningful.
            if not np.isfinite(label):
                signed_move_atr = (close[end_idx] - entry) / max(atr_vals[i], 1e-9)
                if abs(signed_move_atr) < neutral_atr_mult:
                    continue
                label = 1.0 if signed_move_atr > 0 else 0.0
                winner = "vb"

            labels[i] = label
            signed_move_atr = (close[exit_idx] - entry) / max(atr_vals[i], 1e-9)
            vol_adj_move[i] = signed_move_atr
            raw_quality = abs(signed_move_atr)
            if winner == "tp":
                raw_quality += 0.35
            elif winner == "sl":
                raw_quality += 0.10
            raw_quality *= (1.0 + 0.20 * trend_strength - 0.10 * stress)
            quality[i] = float(np.clip(raw_quality, 0.05, 3.0))

        out = df.copy()
        out['ATR_20'] = atr
        out['ADX_14'] = adx14
        out['TB_TP_ATR'] = dynamic_tp
        out['TB_SL_ATR'] = dynamic_sl
        out['TB_Horizon'] = dynamic_hz
        out['TB_VolAdj_Move'] = vol_adj_move
        out['Target_Quality'] = quality
        out['Target'] = labels
        return out

    def process_news(self):
        print(f"[Analyzer] Reading {self.ticker} news from {self.news_path}...")
        
        if not os.path.exists(self.news_path):
            print(f"Error: file not found: {self.news_path}")
            print("Run data collection first.")
            return None

        # Read the CSV file
        df = pd.read_csv(self.news_path)
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'date' not in df.columns or 'title' not in df.columns:
             print(f"Error: CSV is missing date or title columns. Current columns: {df.columns.tolist()}")
             return None

        # Normalize dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.date
        df.dropna(subset=['date'], inplace=True)
        df = df[df['date'] >= self.regime_start].copy()

        print("Calculating sentiment scores...")
        tqdm.pandas(desc="Sentiment Progress")
        df['sentiment'] = df['title'].progress_apply(self._calculate_base_sentiment)

        daily_df = (
            df.groupby('date')
            .agg(
                sentiment_score=('sentiment', 'mean'),
                headline_count=('sentiment', 'size'),
                sentiment_abs=('sentiment', lambda x: float(np.mean(np.abs(x)))),
                sentiment_std=('sentiment', 'std'),
            )
            .reset_index()
        )
        daily_df['sentiment_std'] = daily_df['sentiment_std'].fillna(0.0)
        
        return daily_df

    def apply_adaptive_learning(self, daily_df, window=20):
        print("Calculating adaptive sentiment features...")
        if 'headline_count' not in daily_df.columns:
            daily_df['headline_count'] = 0.0
        if 'sentiment_abs' not in daily_df.columns:
            daily_df['sentiment_abs'] = daily_df['sentiment_score'].abs()
        if 'sentiment_std' not in daily_df.columns:
            daily_df['sentiment_std'] = 0.0

        daily_df['rolling_mean'] = daily_df['sentiment_score'].rolling(window=window, min_periods=5).mean()
        daily_df['rolling_std'] = daily_df['sentiment_score'].rolling(window=window, min_periods=5).std()
        
        daily_df['sentiment_z_score'] = (
            (daily_df['sentiment_score'] - daily_df['rolling_mean']) / daily_df['rolling_std']
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        daily_df['adaptive_score'] = daily_df['sentiment_z_score']
        daily_df['sentiment_ema_fast'] = daily_df['sentiment_score'].ewm(span=3, adjust=False).mean()
        daily_df['sentiment_ema_slow'] = daily_df['sentiment_score'].ewm(span=20, adjust=False).mean()
        daily_df['sentiment_impulse'] = daily_df['sentiment_ema_fast'] - daily_df['sentiment_ema_slow']
        daily_df['sentiment_decay'] = daily_df['sentiment_score'].ewm(halflife=5, adjust=False).mean()

        daily_df['news_intensity'] = np.log1p(pd.to_numeric(daily_df['headline_count'], errors='coerce').fillna(0.0))
        news_roll_mean = daily_df['news_intensity'].rolling(window=window, min_periods=5).mean()
        news_roll_std = daily_df['news_intensity'].rolling(window=window, min_periods=5).std().replace(0, np.nan)
        daily_df['news_intensity_z20'] = (
            (daily_df['news_intensity'] - news_roll_mean) / news_roll_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return daily_df

    def merge_with_price(self, sentiment_df, target_params=None):
        print(f"Merging sentiment with {self.ticker} price history...")
        
        if not os.path.exists(self.stock_path):
            print(f"Error: price file not found: {self.stock_path}")
            print("Run price collection first.")
            return None

        price_df = pd.read_csv(self.stock_path)
        
        # Normalize column names
        price_df.columns = [c.lower().strip() for c in price_df.columns]
        
        # Validate required date column
        if 'date' not in price_df.columns:
            print("Error: date column not found in price file.")
            return None
            
        # Validate required close column
        if 'close' in price_df.columns:
            price_df.rename(columns={'close': 'Close'}, inplace=True)
        elif 'Close' not in price_df.columns:
            print("Error: close column not found in price file.")
            return None

        # Standardize common OHLC names for downstream logic.
        rename_map = {}
        if 'open' in price_df.columns:
            rename_map['open'] = 'Open'
        if 'high' in price_df.columns:
            rename_map['high'] = 'High'
        if 'low' in price_df.columns:
            rename_map['low'] = 'Low'
        if 'volume' in price_df.columns:
            rename_map['volume'] = 'Volume'
        if rename_map:
            price_df.rename(columns=rename_map, inplace=True)

        # Normalize dates
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.date
        
        # Keep all trading days and carry the latest sentiment forward.
        merged_df = pd.merge(price_df, sentiment_df, on='date', how='left')
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        if 'lexicon_score' in merged_df.columns:
            merged_df.drop(columns=['lexicon_score'], inplace=True)
        merged_df['sentiment_score'] = merged_df['sentiment_score'].ffill().fillna(0.0)
        merged_df['adaptive_score'] = merged_df['adaptive_score'].ffill().fillna(0.0)
        if 'sentiment_impulse' not in merged_df.columns:
            merged_df['sentiment_impulse'] = 0.0
        if 'sentiment_decay' not in merged_df.columns:
            merged_df['sentiment_decay'] = merged_df['sentiment_score']
        if 'headline_count' not in merged_df.columns:
            merged_df['headline_count'] = 0.0
        if 'news_intensity_z20' not in merged_df.columns:
            merged_df['news_intensity_z20'] = 0.0
        merged_df['headline_count'] = pd.to_numeric(merged_df['headline_count'], errors='coerce').fillna(0.0)
        merged_df['news_intensity_z20'] = pd.to_numeric(merged_df['news_intensity_z20'], errors='coerce').fillna(0.0)

        news_flag = merged_df['headline_count'] > 0
        idx = np.arange(len(merged_df), dtype=float)
        last_news_idx = pd.Series(np.where(news_flag.to_numpy(), idx, np.nan)).ffill().to_numpy(dtype=float)
        recency = idx - last_news_idx
        recency[~np.isfinite(last_news_idx)] = 999.0
        merged_df['News_Recency_Days'] = recency

        # Add technical features
        merged_df = self._add_technical_indicators(merged_df)

        # Add orthogonal market-context features
        merged_df = self._add_orthogonal_features(merged_df)

        # Strengthen sentiment signal: combine probability with recency-aware impulse.
        sentiment_proxy = np.clip(
            0.5
            + 0.30 * pd.to_numeric(merged_df['adaptive_score'], errors='coerce').fillna(0.0)
            + 0.20 * pd.to_numeric(merged_df['sentiment_impulse'], errors='coerce').fillna(0.0),
            0.0,
            1.0
        )
        if 'FinBERT_Prob' in merged_df.columns:
            merged_df['FinBERT_Prob'] = pd.to_numeric(merged_df['FinBERT_Prob'], errors='coerce')
            merged_df['FinBERT_Prob'] = merged_df['FinBERT_Prob'].fillna(sentiment_proxy)
        else:
            merged_df['FinBERT_Prob'] = sentiment_proxy

        fb_roll_mean = merged_df['FinBERT_Prob'].rolling(20, min_periods=5).mean()
        fb_roll_std = merged_df['FinBERT_Prob'].rolling(20, min_periods=5).std().replace(0, np.nan)
        merged_df['FinBERT_Prob_Z20'] = (
            (merged_df['FinBERT_Prob'] - fb_roll_mean) / fb_roll_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
        merged_df['FinBERT_Prob_5D_Chg'] = (
            merged_df['FinBERT_Prob'].pct_change(periods=5, fill_method=None)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        decay = np.exp(-pd.to_numeric(merged_df['News_Recency_Days'], errors='coerce').fillna(999.0) / 7.0)
        merged_df['FinBERT_Conviction'] = (merged_df['FinBERT_Prob'] - 0.5) * decay

        # Create triple-barrier target labels
        target_params = target_params or {}
        tp_mult = float(target_params.get('tp_atr_mult', 1.8))
        sl_mult = float(target_params.get('sl_atr_mult', 1.4))
        hold_bars = int(target_params.get('max_holding_bars', 12))
        neutral_mult = float(target_params.get('neutral_atr_mult', 0.30))
        merged_df = self._build_triple_barrier_target(
            merged_df,
            atr_window=20,
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
            max_holding_bars=hold_bars,
            neutral_atr_mult=neutral_mult,
        )
        merged_df = merged_df.dropna(subset=['Target']).copy()
        merged_df['Target'] = merged_df['Target'].astype(int)
        merged_df = merged_df[merged_df['date'] >= self.regime_start].copy()
        
        # Keep the final training columns
        cols_to_keep = [
            'date', 'sentiment_score', 'adaptive_score', 
            'headline_count', 'sentiment_abs', 'sentiment_std',
            'sentiment_impulse', 'sentiment_decay', 'news_intensity_z20', 'News_Recency_Days',
            'Trend_Signal', 'RSI', 'Momentum_5D', 
            'Close',
            'Peer_RS_20D', 'SPY_Above_200DMA', 'VIX_Close', 'VIX_5D_Chg', 'Risk_On',
            'RS_vs_QQQ_20D', 'RS_vs_SOXX_20D', 'QQQ_Above_50DMA',
            'HV_20_Ann', 'VIX_Z20',
            'PutCall_Total', 'PutCall_Equity', 'PutCall_Index',
            'PutCall_EquityMinusIndex', 'PutCall_Z20', 'PutCall_5D_Chg',
            'FinBERT_Prob', 'FinBERT_Prob_Z20', 'FinBERT_Prob_5D_Chg', 'FinBERT_Conviction',
            'FracDiff_Close_d04', 'FracDiff_Close_d07', 'FracDiff_VIX_d04',
            'CMF_20', 'OBV_Z20', 'DollarVol_Z20',
            'SC_TSM_Sent', 'SC_ASML_Sent', 'SC_SMCI_Sent',
            'SC_MSFT_CAPEX_Sent', 'SC_GOOGL_CAPEX_Sent', 'SC_AMZN_CAPEX_Sent',
            'SupplyChain_Sent',
            'TB_TP_ATR', 'TB_SL_ATR', 'TB_Horizon', 'TB_VolAdj_Move', 'Target_Quality',
            'Target'
        ]
        final_cols = [c for c in cols_to_keep if c in merged_df.columns]
        
        return merged_df[final_cols]

    def run(self, target_params=None):
        daily_df = self.process_news()
        if daily_df is None: return

        learned_df = self.apply_adaptive_learning(daily_df)
        learned_df.to_csv(self.output_sentiment_path, index=False)
        final_df = self.merge_with_price(learned_df, target_params=target_params)
        
        if final_df is not None and not final_df.empty:
            final_df.to_csv(self.final_merged_path, index=False)
            print(f"Training dataset created: {self.final_merged_path}")
            if target_params:
                print(f"Target params used: {target_params}")
            print("Included fields: sentiment, technical features, and target labels")
        else:
            print("Warning: merged dataset is empty. Check whether the news and price dates overlap.")

# ==========================================
# 3. Public entry point for main.py
# ==========================================
def run_sentiment_analysis(ticker="META", target_params=None, regime_start="2014-01-01"):
    # Instantiate and run the analyzer.
    analyzer = AdaptiveSentimentAnalyzer(ticker=ticker, regime_start=regime_start)
    analyzer.run(target_params=target_params)

if __name__ == "__main__":
    run_sentiment_analysis("META")
