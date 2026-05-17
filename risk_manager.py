"""
risk_manager.py
===============
ניהול סיכונים דינמי: מגביל גודל פוזיציה בהתאם ל-drawdown,
ומפעיל התראות ועצירה אוטומטית.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

from enum import Enum
import logging

import numpy as np

logger = logging.getLogger("RiskManager")

# ─── רמות סיכון ───────────────────────────────────────────────────────────────
DRAWDOWN_REDUCE  = 0.10   # drawdown של 10% → הקטנת פוזיציה ל-50%
DRAWDOWN_HALT    = 0.15   # drawdown של 15% → עצירה מלאה
POSITION_NORMAL  = 1.0    # מכפיל פוזיציה רגיל
POSITION_REDUCED = 0.5    # מכפיל פוזיציה מצומצם


class RiskLevel(Enum):
    NORMAL  = "NORMAL"   # מסחר רגיל
    REDUCED = "REDUCED"  # פוזיציה מוקטנת
    HALTED  = "HALTED"   # עצירה מלאה


class RiskManager:
    """
    מנהל סיכונים דינמי.

    - עוקב אחרי peak value ו-drawdown שוטף.
    - ב-10% drawdown: מצמצם פקודות ל-50% גודל.
    - ב-15% drawdown: מפסיק מסחר ומדפיס התראה.
    """

    def __init__(
        self,
        initial_capital: float,
        drawdown_reduce: float = DRAWDOWN_REDUCE,
        drawdown_halt: float = DRAWDOWN_HALT,
    ):
        self.initial_capital = initial_capital
        self.drawdown_reduce = drawdown_reduce
        self.drawdown_halt   = drawdown_halt

        self.peak_value        = initial_capital
        self.risk_level        = RiskLevel.NORMAL
        self._current_drawdown = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # API ציבורי
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, current_value: float) -> RiskLevel:
        """
        מעדכן רמת סיכון בהתאם לשווי הנוכחי.
        מחזיר את רמת הסיכון הפעילה.
        """
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

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        מכפיל את גודל הפעולה לפי רמת הסיכון.
        במצב HALTED – מחזיר פעולות אפס (אין מסחר).
        """
        if self.risk_level == RiskLevel.HALTED:
            return np.zeros_like(action)

        if self.risk_level == RiskLevel.REDUCED:
            # מצמצם רק קניות (ערכים חיוביים)
            scaled = action.copy()
            scaled[scaled > 0] *= POSITION_REDUCED
            return scaled

        return action  # NORMAL – ללא שינוי

    @property
    def is_halted(self) -> bool:
        return self.risk_level == RiskLevel.HALTED

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    def get_status(self) -> dict:
        return {
            "risk_level":      self.risk_level.value,
            "peak_value":      self.peak_value,
            "current_drawdown": self._current_drawdown,
            "is_halted":       self.is_halted,
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
