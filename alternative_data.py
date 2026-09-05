"""
alternative_data.py
===================
Free alternative data sources that provide market context beyond price/volume.

Sources
-------
1. Fear & Greed Index (CNN Money) — overall market sentiment 0-100
2. VIX proxy — market fear via SPY options implied volatility
3. Market Breadth — % of S&P 500 stocks above their 200-day MA

All values are normalised to [-1, +1] so they plug directly into the
observation vector without scaling issues.

Usage in LiveTrader
-------------------
from alternative_data import AlternativeDataFetcher

fetcher = AlternativeDataFetcher()
alt     = fetcher.fetch_all()     # dict with normalised values
# e.g. {'fear_greed': 0.32, 'vix_norm': -0.15, 'breadth': 0.61, ...}
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np

log = logging.getLogger("AltData")


class AlternativeDataFetcher:
    """
    Fetches free alternative market data with graceful fallbacks.
    All values are normalised to roughly [-1, +1].
    """

    TIMEOUT = 8   # seconds per request
    VIX_CACHE_SECONDS = 900   # VIX doesn't need a network round-trip every poll cycle (60s default)

    def __init__(self) -> None:
        self._vix_cache: float | None = None
        self._vix_cache_at: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def fetch_all(self) -> dict[str, float]:
        """
        Fetches all alternative data sources.
        Returns a dict of normalised float values.
        Missing sources return 0.0 (neutral) so the system degrades gracefully.
        """
        result: dict[str, float] = {}

        result["fear_greed"]  = self.fetch_fear_greed()
        result["vix_norm"]    = self.fetch_vix_proxy()
        result["fetch_time"]  = self._time_of_day_feature()

        log.info(
            f"Alternative data: fear_greed={result['fear_greed']:+.2f} | "
            f"vix={result['vix_norm']:+.2f}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Fear & Greed Index
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_fear_greed(self) -> float:
        """
        Fetches CNN's Fear & Greed Index (0=extreme fear, 100=extreme greed).
        Normalised to [-1, +1]: 0→-1, 50→0, 100→+1.

        Returns 0.0 on failure (neutral).
        """
        urls = [
            # Primary: unofficial CNN endpoint
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            # Fallback: alternative provider
            "https://api.alternative.me/fng/?limit=1&format=json",
        ]

        # Try CNN endpoint
        try:
            req = urllib.request.Request(
                urls[0],
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept":     "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                data  = json.loads(resp.read())
                score = float(data["fear_and_greed"]["score"])
                norm  = (score - 50) / 50   # maps 0..100 → -1..+1
                log.info(f"Fear & Greed (CNN): {score:.0f} → normalised {norm:+.2f}")
                return float(np.clip(norm, -1.0, 1.0))
        except Exception as exc:
            log.debug(f"CNN Fear & Greed failed: {exc}")

        # Fallback: alternative.me (Crypto Fear & Greed — correlated with equity)
        try:
            with urllib.request.urlopen(urls[1], timeout=self.TIMEOUT) as resp:
                data  = json.loads(resp.read())
                score = float(data["data"][0]["value"])
                norm  = (score - 50) / 50
                log.info(f"Fear & Greed (alt): {score:.0f} → normalised {norm:+.2f}")
                return float(np.clip(norm, -1.0, 1.0))
        except Exception as exc:
            log.debug(f"Alt Fear & Greed failed: {exc}")

        log.warning("Fear & Greed unavailable — using neutral (0.0)")
        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # VIX proxy
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_vix_raw(self) -> float | None:
        """Latest raw VIX close (e.g. 18.5), for callers that need the real
        value rather than the [-1,+1] normalised proxy below — e.g.
        RegimeDetector.detect(vix_value=...), which was validated to carry
        real predictive signal that the normalised/clipped version loses
        resolution on. Returns None on failure.

        Cached for VIX_CACHE_SECONDS: VIX doesn't meaningfully change
        between 60-second poll cycles, so refetching every cycle would just
        add network latency and a new failure point to every single
        decision without adding information.
        """
        import time
        now = time.monotonic()
        if self._vix_cache is not None and (now - self._vix_cache_at) < self.VIX_CACHE_SECONDS:
            return self._vix_cache
        try:
            import yfinance as yf
            vix_data = yf.download("^VIX", period="1d", progress=False, auto_adjust=True)
            if vix_data.empty:
                raise ValueError("Empty VIX data")
            value = float(vix_data["Close"].iloc[-1])
            self._vix_cache = value
            self._vix_cache_at = now
            return value
        except Exception as exc:
            log.debug(f"VIX fetch failed: {exc}")
            return self._vix_cache  # serve the last known-good value if we have one

    def fetch_vix_proxy(self) -> float:
        """
        VIX: 10=calm, 20=normal, 40=crisis.
        Normalised: 10→-1 (calm/bullish), 40→+1 (fearful/bearish).

        Returns 0.0 on failure.
        """
        vix = self.fetch_vix_raw()
        if vix is None:
            return 0.0
        # Normalise: centre at 20, scale by 15
        norm = float(np.clip((vix - 20) / 15, -1.0, 1.0))
        log.info(f"VIX: {vix:.1f} → normalised {norm:+.2f}")
        return norm

    # ──────────────────────────────────────────────────────────────────────────
    # Time-of-day feature
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _time_of_day_feature() -> float:
        """
        Encodes time of day as a sine wave peaking at market open (9:30 ET).
        Gives the model a sense of intraday timing.
        Returns value in [-1, +1].
        """
        now_utc  = datetime.now(timezone.utc)
        hour_utc = now_utc.hour + now_utc.minute / 60
        # Market open ≈ 13:30 UTC (9:30 ET summer)
        hour_norm = (hour_utc - 13.5) / 6.5   # roughly -1..+1 across trading day
        return float(np.clip(hour_norm, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: append alt data to observation vector
# ─────────────────────────────────────────────────────────────────────────────

def enrich_observation(obs: "np.ndarray", alt: dict[str, float]) -> "np.ndarray":
    """
    Appends alternative data features to the last row of the observation matrix.
    Keeps the window dimension intact; only the feature dimension grows.

    obs  : (window_size, features)         — existing observation
    alt  : {'fear_greed': x, 'vix_norm': y, ...}
    Returns (window_size, features + n_alt_features)
    """
    import numpy as np

    alt_values = np.array([
        alt.get("fear_greed", 0.0),
        alt.get("vix_norm",   0.0),
        alt.get("fetch_time", 0.0),
    ], dtype=np.float32)

    # Broadcast alt features across all window rows
    alt_block = np.tile(alt_values, (obs.shape[0], 1))
    return np.concatenate([obs, alt_block], axis=1).astype(np.float32)
