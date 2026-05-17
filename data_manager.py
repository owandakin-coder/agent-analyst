"""
data_manager.py
===============
מנהל הנתונים – אחראי על הורדת, עיבוד ושמירת נתוני שוק.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── קבועים ───────────────────────────────────────────────────────────────────
CACHE_DIR = "cache"
DEFAULT_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META",
    # EV / Consumer
    "TSLA",
    # Finance
    "JPM", "V", "BAC",
    # Healthcare
    "JNJ", "UNH",
    # Energy
    "XOM",
    # Retail
    "WMT",
    # Index ETF
    "SPY",
]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"
RSI_PERIOD = 14
MA_SHORT = 20
MA_LONG = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_PERIOD = 20
BOLL_STD = 2
ATR_PERIOD = 14


class DataManager:
    """
    מנהל נתוני שוק: הורדה, חישוב פיצ'רים, מטמון.
    """

    def __init__(
        self,
        tickers: list[str] = DEFAULT_TICKERS,
        start: str = DEFAULT_START,
        end: str = DEFAULT_END,
        cache_dir: str = CACHE_DIR,
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # מילון: ticker -> DataFrame עם כל הפיצ'רים
        self.data: dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # API ציבורי
    # ──────────────────────────────────────────────────────────────────────────

    def load_all(self, force_download: bool = False) -> dict[str, pd.DataFrame]:
        """טוען את כל המניות (ממטמון אם קיים, אחרת מוריד)."""
        if force_download:
            self._cleanup_stale_cache()
        for ticker in self.tickers:
            self.data[ticker] = self._load_ticker(ticker, force_download)
        print(f"[DataManager] Loaded {len(self.data)} tickers: {self.tickers}")
        return self.data

    def _cleanup_stale_cache(self, keep_latest: int = 3):
        """מוחק קבצי cache ישנים לכל ticker, שומר רק את ה-keep_latest האחרונים."""
        import glob as _glob
        for ticker in self.tickers:
            pattern = os.path.join(self.cache_dir, f"{ticker}_*.pkl")
            files   = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
            for old_file in files[keep_latest:]:
                try:
                    os.remove(old_file)
                    print(f"[DataManager] Removed stale cache: {old_file}")
                except OSError:
                    pass

    def get_aligned_data(self) -> dict[str, pd.DataFrame]:
        """מחזיר נתונים מסונכרנים לאותן תאריכים (inner join על ציר הזמן)."""
        if not self.data:
            self.load_all()

        # מוצא תאריכים משותפים לכל המניות
        common_idx = None
        for df in self.data.values():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)

        aligned = {}
        for ticker, df in self.data.items():
            aligned[ticker] = df.loc[common_idx].copy()

        print(f"[DataManager] {len(common_idx)} common trading days.")
        return aligned

    def get_feature_names(self) -> list[str]:
        """מחזיר רשימת שמות הפיצ'רים."""
        if not self.data:
            raise RuntimeError("יש לקרוא load_all() תחילה.")
        first_df = next(iter(self.data.values()))
        return list(first_df.columns)

    # ──────────────────────────────────────────────────────────────────────────
    # לוגיקה פנימית
    # ──────────────────────────────────────────────────────────────────────────

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_{self.start}_{self.end}.pkl")

    def _load_ticker(self, ticker: str, force_download: bool) -> pd.DataFrame:
        cache_file = self._cache_path(ticker)

        if not force_download and os.path.exists(cache_file):
            print(f"[DataManager] {ticker}: loading from cache ({cache_file})")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print(f"[DataManager] {ticker}: downloading from yfinance ...")
        raw = yf.download(ticker, start=self.start, end=self.end, progress=False)

        if raw.empty:
            raise ValueError(f"לא התקבלו נתונים עבור {ticker}")

        # מטפח עמודות multi-level אם קיימות
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = self._compute_features(raw, ticker)

        # שמירה למטמון
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        print(f"[DataManager] {ticker}: saved to cache.")
        return df

    def _compute_features(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """מחשב את כל הפיצ'רים הטכניים על OHLCV גולמי."""
        df = pd.DataFrame(index=raw.index)

        close = raw["Close"].squeeze()
        high  = raw["High"].squeeze()
        low   = raw["Low"].squeeze()
        volume = raw["Volume"].squeeze()

        # ── מחירים בסיסיים ─────────────────────────────────────────────────
        df["close"]  = close
        df["open"]   = raw["Open"].squeeze()
        df["high"]   = high
        df["low"]    = low
        df["volume"] = volume

        # ── תשואות יומיות ──────────────────────────────────────────────────
        df["returns"]     = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # ── ממוצעים נעים ───────────────────────────────────────────────────
        df["ma20"] = close.rolling(MA_SHORT).mean()
        df["ma50"] = close.rolling(MA_LONG).mean()

        # יחס מחיר/ממוצע (מנורמל)
        df["price_to_ma20"] = close / df["ma20"] - 1
        df["price_to_ma50"] = close / df["ma50"] - 1
        df["ma_cross"]      = df["ma20"] / df["ma50"] - 1  # חציית ממוצעים

        # ── RSI ────────────────────────────────────────────────────────────
        df["rsi"] = self._rsi(close, RSI_PERIOD)

        # ── MACD ───────────────────────────────────────────────────────────
        ema_fast   = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["macd"]         = macd_line
        df["macd_signal"]  = signal_line
        df["macd_hist"]    = macd_line - signal_line  # היסטוגרמה

        # ── Bollinger Bands ────────────────────────────────────────────────
        boll_mid = close.rolling(BOLL_PERIOD).mean()
        boll_std = close.rolling(BOLL_PERIOD).std()
        df["boll_upper"] = boll_mid + BOLL_STD * boll_std
        df["boll_lower"] = boll_mid - BOLL_STD * boll_std
        # מיקום יחסי בתוך הרצועה [0,1]
        boll_range = df["boll_upper"] - df["boll_lower"]
        df["boll_pct"] = (close - df["boll_lower"]) / boll_range.replace(0, np.nan)
        df["boll_width"] = boll_range / boll_mid  # רוחב מנורמל

        # ── ATR (Average True Range) ───────────────────────────────────────
        df["atr"] = self._atr(high, low, close, ATR_PERIOD)
        df["atr_pct"] = df["atr"] / close  # ATR יחסי

        # ── שינוי נפח ──────────────────────────────────────────────────────
        df["volume_change"]  = volume.pct_change()
        df["volume_ma20"]    = volume.rolling(MA_SHORT).mean()
        df["volume_ratio"]   = volume / df["volume_ma20"]  # נפח יחסי

        # ── תנודתיות היסטורית ──────────────────────────────────────────────
        df["volatility_20"] = df["returns"].rolling(MA_SHORT).std() * np.sqrt(252)

        # ── נורמליזציה פשוטה: z-score על חלון גלגל ────────────────────────
        for col in ["returns", "log_returns", "rsi", "macd_hist", "boll_pct",
                    "volume_ratio", "atr_pct", "volatility_20"]:
            roll_mean = df[col].rolling(252, min_periods=60).mean()
            roll_std  = df[col].rolling(252, min_periods=60).std()
            df[f"{col}_z"] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)

        # ── ניקוי ערכים חסרים ──────────────────────────────────────────────
        df.dropna(inplace=True)

        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """חישוב RSI (Relative Strength Index)."""
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """חישוב ATR (Average True Range)."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()
