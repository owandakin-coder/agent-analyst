"""
risk_manager.py
===============
Dynamic risk management:
- Drawdown-based position scaling and halt
- Kelly Criterion for optimal position sizing
- Correlation-aware position limits (avoid over-concentration in correlated assets)
- Regime-aware multiplier (integrates with RegimeDetector)

For research purposes only.
"""

from __future__ import annotations

from enum import Enum
import logging

import numpy as np

logger = logging.getLogger("RiskManager")

# ─── Load from config (fallback to hardcoded if config not available) ─────────
try:
    from config_loader import CFG
    DRAWDOWN_REDUCE  = CFG.drawdown_reduce
    DRAWDOWN_HALT    = CFG.drawdown_halt
    POSITION_NORMAL  = CFG.get("risk", "position_normal",  default=1.0)
    POSITION_REDUCED = CFG.get("risk", "position_reduced", default=0.5)
    KELLY_FRACTION   = CFG.kelly_fraction
    KELLY_MIN        = CFG.kelly_min
    KELLY_MAX        = CFG.kelly_max
    CORR_HIGH        = CFG.corr_threshold
    CORR_WINDOW      = CFG.corr_window
except Exception:
    # Fallback defaults
    DRAWDOWN_REDUCE  = 0.10
    DRAWDOWN_HALT    = 0.15
    POSITION_NORMAL  = 1.0
    POSITION_REDUCED = 0.5
    KELLY_FRACTION   = 0.25
    KELLY_MIN        = 0.10
    KELLY_MAX        = 1.00
    CORR_HIGH        = 0.80
    CORR_WINDOW      = 60


class RiskLevel(Enum):
    NORMAL  = "NORMAL"
    REDUCED = "REDUCED"
    HALTED  = "HALTED"


class RiskManager:
    """
    Dynamic risk manager combining:
    1. Drawdown-based halt / reduce
    2. Kelly Criterion for optimal position sizing
    3. Correlation-aware scaling (avoid piling into correlated assets)
    4. Regime multiplier from RegimeDetector

    Parameters
    ----------
    initial_capital : float
    drawdown_reduce : float   — drawdown level that triggers REDUCED mode
    drawdown_halt   : float   — drawdown level that triggers HALTED mode
    use_kelly       : bool    — enable Kelly Criterion sizing
    use_correlation : bool    — enable correlation-aware scaling
    """

    def __init__(
        self,
        initial_capital: float,
        drawdown_reduce: float = DRAWDOWN_REDUCE,
        drawdown_halt:   float = DRAWDOWN_HALT,
        use_kelly:       bool  = True,
        use_correlation: bool  = True,
    ):
        self.initial_capital = initial_capital
        self.drawdown_reduce = drawdown_reduce
        self.drawdown_halt   = drawdown_halt
        self.use_kelly       = use_kelly
        self.use_correlation = use_correlation

        self.peak_value        = initial_capital
        self.risk_level        = RiskLevel.NORMAL
        self._current_drawdown = 0.0

        # Kelly tracking: rolling trade outcomes per ticker
        self._trade_outcomes: dict[str, list[float]] = {}   # ticker → [+1/-1, ...]

        # Regime multiplier (set externally by RegimeDetector)
        self._regime_multiplier: float = 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, current_value: float) -> RiskLevel:
        """Updates drawdown state. Returns current RiskLevel."""
        self.peak_value = max(self.peak_value, current_value)
        drawdown = self._compute_drawdown(current_value)
        self._current_drawdown = drawdown

        if drawdown >= self.drawdown_halt:
            self._activate_halt(drawdown)
        elif drawdown >= self.drawdown_reduce:
            self._activate_reduced(drawdown)
        else:
            self._activate_normal(drawdown)

        return self.risk_level

    def set_regime_multiplier(self, multiplier: float):
        """
        Called by LiveTrader after RegimeDetector runs.
        multiplier: 1.0=Bull, 0.6=Sideways, 0.3=Bear
        """
        self._regime_multiplier = float(np.clip(multiplier, 0.0, 1.0))
        logger.info(f"Regime multiplier set to {self._regime_multiplier:.2f}")

    def scale_action(
        self,
        action: np.ndarray,
        tickers: list[str] | None = None,
        price_history: dict[str, "pd.Series"] | None = None,
    ) -> np.ndarray:
        """
        Applies all risk layers to the raw action vector.

        Layers applied in order:
        1. HALTED  → zeros
        2. REDUCED → buy signals halved
        3. Regime  → multiply by regime multiplier
        4. Kelly   → per-ticker optimal sizing
        5. Correlation → reduce correlated clusters

        Parameters
        ----------
        action        : raw action vector from model, shape (n_stocks,)
        tickers       : ticker names corresponding to action indices
        price_history : {ticker: pd.Series of close prices} for correlation
        """
        if self.risk_level == RiskLevel.HALTED:
            return np.zeros_like(action)

        scaled = action.copy()

        # Layer 2: REDUCED mode
        if self.risk_level == RiskLevel.REDUCED:
            scaled[scaled > 0] *= POSITION_REDUCED

        # Layer 3: Regime multiplier
        if self._regime_multiplier < 1.0:
            scaled[scaled > 0] *= self._regime_multiplier

        # Layer 4: Kelly Criterion
        if self.use_kelly and tickers:
            scaled = self._apply_kelly(scaled, tickers)

        # Layer 5: Correlation-aware scaling
        if self.use_correlation and tickers and price_history:
            scaled = self._apply_correlation_scaling(scaled, tickers, price_history)

        return np.clip(scaled, -1.0, 1.0)

    def record_trade_outcome(self, ticker: str, pnl_pct: float):
        """
        Records a trade outcome for Kelly Criterion tracking.
        Call this after each completed trade with the P&L percentage.
        """
        if ticker not in self._trade_outcomes:
            self._trade_outcomes[ticker] = []
        self._trade_outcomes[ticker].append(pnl_pct)
        # Keep last 100 trades only
        self._trade_outcomes[ticker] = self._trade_outcomes[ticker][-100:]

    @property
    def is_halted(self) -> bool:
        return self.risk_level == RiskLevel.HALTED

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    def get_status(self) -> dict:
        return {
            "risk_level":        self.risk_level.value,
            "peak_value":        self.peak_value,
            "current_drawdown":  self._current_drawdown,
            "is_halted":         self.is_halted,
            "regime_multiplier": self._regime_multiplier,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # פנימי
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_drawdown(self, current_value: float) -> float:
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - current_value) / self.peak_value

    def _activate_halt(self, drawdown: float):
        if self.risk_level != RiskLevel.HALTED:
            self.risk_level = RiskLevel.HALTED
            logger.warning(
                f"🚨 TRADING HALTED! Drawdown={drawdown:.1%} "
                f"(threshold: {self.drawdown_halt:.0%}). "
                "Manual restart required — or drawdown must recover below 10%."
            )

    def _activate_reduced(self, drawdown: float):
        if self.risk_level == RiskLevel.HALTED:
            # Recovery from HALT: only when drawdown drops below reduce threshold
            logger.info(
                f"✅ HALT LIFTED — Drawdown recovered to {drawdown:.1%} "
                f"(below {self.drawdown_reduce:.0%}). Switching to REDUCED."
            )
        if self.risk_level != RiskLevel.REDUCED:
            self.risk_level = RiskLevel.REDUCED
            logger.warning(
                f"⚠️ POSITION SIZE REDUCED to 50%. Drawdown={drawdown:.1%} "
                f"(threshold: {self.drawdown_reduce:.0%})."
            )

    def _activate_normal(self, drawdown: float):
        if self.risk_level != RiskLevel.NORMAL:
            prev = self.risk_level
            self.risk_level = RiskLevel.NORMAL
            if prev == RiskLevel.HALTED:
                logger.info(f"✅ HALT LIFTED — Drawdown fully recovered to {drawdown:.1%}. NORMAL trading resumed.")
            else:
                logger.info(f"✅ Risk level back to NORMAL. Drawdown={drawdown:.1%}")

    # ──────────────────────────────────────────────────────────────────────────
    # Kelly Criterion
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_kelly(self, action: np.ndarray, tickers: list[str]) -> np.ndarray:
        """
        Scales each buy signal by the fractional Kelly multiplier derived from
        that ticker's historical trade outcomes.

        Kelly formula:
            f* = (p * b - q) / b
        where:
            p = win rate (fraction of profitable trades)
            q = 1 - p
            b = average win / average loss (profit factor)

        We use KELLY_FRACTION * f* so we never bet the full Kelly amount.
        Falls back to 1.0 (no change) if fewer than 10 recorded trades.
        """
        scaled = action.copy()
        for i, ticker in enumerate(tickers):
            if scaled[i] <= 0:
                continue   # only scale buy signals

            outcomes = self._trade_outcomes.get(ticker, [])
            if len(outcomes) < 10:
                continue   # not enough data → leave unchanged

            arr = np.array(outcomes)
            wins  = arr[arr > 0]
            losses = arr[arr < 0]

            if len(wins) == 0 or len(losses) == 0:
                continue

            p = len(wins) / len(arr)
            q = 1.0 - p
            avg_win  = float(wins.mean())
            avg_loss = float(np.abs(losses).mean())

            if avg_loss == 0:
                continue

            b    = avg_win / avg_loss
            kelly = (p * b - q) / b
            kelly = float(np.clip(kelly * KELLY_FRACTION, KELLY_MIN, KELLY_MAX))

            scaled[i] *= kelly
            logger.debug(
                f"Kelly {ticker}: p={p:.0%} b={b:.2f} → f={kelly:.2f} "
                f"(signal {action[i]:.2f} → {scaled[i]:.2f})"
            )

        return scaled

    # ──────────────────────────────────────────────────────────────────────────
    # Correlation-aware scaling
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_correlation_scaling(
        self,
        action: np.ndarray,
        tickers: list[str],
        price_history: "dict[str, pd.Series]",
    ) -> np.ndarray:
        """
        If two buy signals belong to assets with rolling correlation > CORR_HIGH,
        the smaller signal is halved to avoid concentration risk.

        Algorithm:
        1. Build a correlation matrix from the last CORR_WINDOW days of returns.
        2. For each pair (i, j) where corr > CORR_HIGH and both signals > 0:
           - Keep the larger signal intact; halve the smaller one.

        Only active buy positions are affected; sells and holds are untouched.
        """
        try:
            import pandas as pd
        except ImportError:
            return action

        scaled = action.copy()

        # Collect tickers with active buy signals
        buy_indices = [i for i, a in enumerate(scaled) if a > 0 and tickers[i] in price_history]
        if len(buy_indices) < 2:
            return scaled   # nothing to correlate

        # Build returns DataFrame
        series_list = {}
        for i in buy_indices:
            t = tickers[i]
            s = price_history[t]
            if hasattr(s, "iloc") and len(s) >= CORR_WINDOW + 1:
                series_list[t] = s.iloc[-CORR_WINDOW - 1:].pct_change().dropna()

        if len(series_list) < 2:
            return scaled

        ret_df = pd.DataFrame(series_list).dropna()
        if ret_df.empty or ret_df.shape[0] < 10:
            return scaled

        corr_matrix = ret_df.corr()

        # For each high-correlation pair, penalise the weaker signal
        penalised: set[int] = set()
        for ii, i in enumerate(buy_indices):
            for jj, j in enumerate(buy_indices):
                if jj <= ii:
                    continue
                ti, tj = tickers[i], tickers[j]
                if ti not in corr_matrix.columns or tj not in corr_matrix.columns:
                    continue
                corr = corr_matrix.loc[ti, tj]
                if corr > CORR_HIGH:
                    # Keep the stronger signal; penalise the weaker
                    if scaled[i] >= scaled[j]:
                        weaker = j
                    else:
                        weaker = i
                    if weaker not in penalised:
                        penalised.add(weaker)
                        scaled[weaker] *= 0.5
                        logger.debug(
                            f"Correlation penalty: {tickers[weaker]} reduced 50% "
                            f"(corr({ti},{tj})={corr:.2f} > {CORR_HIGH})"
                        )

        return scaled
