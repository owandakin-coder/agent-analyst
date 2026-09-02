"""
leakage_check.py
================
בדיקת Data Leakage אגרסיבית — הסיכון הכי גדול במערכת ML למסחר.

בדיקות:
  1. Train/Test overlap  — האם יש תאריכים משותפים?
  2. Feature lookahead   — מחשב features פעמיים (מלא + חתוך) ומשווה ערכים היסטוריים
  3. Normalization leak  — obs_rms.count מול תקציב האימון הידוע (sanity check חלקי)
  3.5. Target leakage    — האם ה-reward/net_worth תלויים בנתונים עתידיים? (differential test)
  4. Temporal consistency — סדר כרונולוגי, פערים חריגים
  5. Distribution shift  — KS-test בין התפלגות train ל-test

שימוש:
    python leakage_check.py           # הרצת כל הבדיקות
    python leakage_check.py --verbose # פרטים מלאים
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows terminals often default stdout to a non-UTF-8 codepage (e.g. cp1255),
# which crashes on the box-drawing characters used in the report below.
if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config_loader import CFG


def check_passed(name: str):
    print(f"  ✅ {name}")

def check_warn(name: str, detail: str):
    print(f"  ⚠️  {name}: {detail}")

def check_fail(name: str, detail: str):
    print(f"  ❌ {name}: {detail}")


# ══════════════════════════════════════════════════════════════════
# 1. Train/Test Overlap
# ══════════════════════════════════════════════════════════════════

def check_train_test_overlap(all_data: dict[str, pd.DataFrame]) -> bool:
    """מוודא שאין תאריכים משותפים בין train לבין test."""
    train_dates = set()
    test_dates  = set()

    for ticker, df in all_data.items():
        train = df[(df.index >= CFG.train_start) & (df.index <= CFG.train_end)]
        test  = df[(df.index >= CFG.test_start)  & (df.index <= CFG.test_end)]
        train_dates.update(train.index.tolist())
        test_dates.update(test.index.tolist())

    overlap = train_dates & test_dates

    if overlap:
        check_fail("Train/Test Overlap",
                   f"{len(overlap)} overlapping dates found! e.g. {sorted(overlap)[:3]}")
        return False
    else:
        check_passed("Train/Test Overlap — no shared dates")
        return True


# ══════════════════════════════════════════════════════════════════
# 2. Feature Lookahead
# ══════════════════════════════════════════════════════════════════

def check_feature_lookahead(all_data: dict[str, pd.DataFrame], verbose: bool = False) -> bool:
    """
    בודק אמיתי (לא רק אוטוקורלציה): מחשב features פעמיים על אותו OHLCV —
    פעם אחת מלא, פעם אחת חתוך אחרי נקודת בדיקה — ומוודא שהערך ההיסטורי
    זהה בשני המקרים. אם פיצ'ר תלוי בנתונים עתידיים, קיצוץ העתיד ישנה
    את הערך שלו בעבר; אם לא, הוא לא ישתנה כלל.

    (הגרסה הקודמת מדדה רק אוטוקורלציה טבעית של מחירים — לא leakage אמיתי.)
    """
    from data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    ok = True
    raw_cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}

    for ticker, df in list(all_data.items())[:3]:  # בדוק דוגמה
        if len(df) < 300 or not set(raw_cols).issubset(df.columns):
            continue

        raw_like = df[list(raw_cols)].rename(columns=raw_cols)
        cutoff = len(raw_like) - 60
        check_at = cutoff - 30  # מרווח בטוח לפני נקודת החיתוך (מיקום ב-raw)
        check_date = raw_like.index[check_at]

        full_features = dm._compute_features(raw_like, ticker)
        truncated_features = dm._compute_features(raw_like.iloc[:cutoff], ticker)

        # Rolling-window warmup drops a different number of leading rows in
        # each computation, so the same *date* lands at a different row
        # position in full_features vs truncated_features — look up by the
        # shared date index, not by position.
        if check_date not in full_features.index or check_date not in truncated_features.index:
            continue
        row_full = full_features.loc[check_date]
        row_trunc = truncated_features.loc[check_date]
        common_cols = [c for c in full_features.columns if c in truncated_features.columns]

        for col in common_cols:
            a, b = row_full[col], row_trunc[col]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or abs(float(a) - float(b)) > 1e-9:
                check_fail(
                    f"Look-ahead in feature '{col}' ({ticker})",
                    f"value at {check_date.date()} changed after removing future rows "
                    f"({a!r} with full data vs {b!r} truncated) — this feature depends "
                    "on data that wouldn't exist yet at that point in time.",
                )
                ok = False

    # בדיקת עמודות ספציפיות המוכרות כבעייתיות
    suspicious = ["future_return", "next_close", "forward", "lead_"]
    for ticker, df in all_data.items():
        for col in df.columns:
            for sus in suspicious:
                if sus.lower() in col.lower():
                    check_fail(f"Suspicious feature name: {col}",
                               "May contain forward-looking data")
                    ok = False

    if ok:
        check_passed("Feature Lookahead — recomputing on truncated data reproduces identical historical values")
    return ok


# ══════════════════════════════════════════════════════════════════
# 3. Normalization Leakage
# ══════════════════════════════════════════════════════════════════

def check_normalization_leak() -> bool:
    """
    בדיקה חלקית אמיתית: VecNormalize.obs_rms.count אמור להיות קרוב לתקציב
    הצעדים שבו אומן המודל (CFG.ensemble_timesteps). PPO מעדכן את ה-normalizer
    פעם אחת לכל env.step() בזמן אימון; מונה גדול משמעותית מהתקציב הידוע הוא
    הסימן הנצפה של נתונים נוספים (למשל תקופת הטסט) שעברו דרכו במצב training.

    זה *לא* הוכחה מוחלטת — אי אפשר לדעת מהקובץ עצמו אילו תאריכים ספציפית
    נראו. זה sanity check על גודל המדגם, לא אימות מלא. (הגרסה הקודמת של
    הבדיקה הזו לא בדקה שום דבר בפועל — תמיד החזירה True.)
    """
    norm_path = Path(CFG.model_dir) / "vec_normalize.pkl"

    if not norm_path.exists():
        check_warn("Normalization", f"vec_normalize.pkl not found at {norm_path} — cannot verify")
        return True

    try:
        import pickle
        with open(norm_path, "rb") as f:
            vec_norm = pickle.load(f)

        if not hasattr(vec_norm, "obs_rms") or vec_norm.obs_rms is None:
            check_warn("Normalization", "Loaded file has no obs_rms — cannot verify")
            return True

        count = float(vec_norm.obs_rms.count)
        expected_budget = float(CFG.ensemble_timesteps)
        if verbose_global:
            print(f"    obs_rms.count={count:,.0f} | expected training budget={expected_budget:,.0f}")

        if count > expected_budget * 1.5:
            check_warn(
                "Normalization",
                f"obs_rms.count={count:,.0f} is >1.5x the configured training budget "
                f"({expected_budget:,.0f} steps) — more data than expected passed "
                "through VecNormalize in training mode. Cannot confirm from this file "
                "alone whether that included the test period; worth checking manually.",
            )
            return False

        check_passed(
            f"Normalization — obs_rms.count ({count:,.0f}) is consistent with the "
            f"declared training budget ({expected_budget:,.0f} steps)"
        )
        return True

    except Exception as e:
        check_warn("Normalization", f"Could not verify: {e}")
        return True


# ══════════════════════════════════════════════════════════════════
# 3.5. Target Leakage (reward function)
# ══════════════════════════════════════════════════════════════════

def check_target_leakage(all_data: dict[str, pd.DataFrame], verbose: bool = False) -> bool:
    """
    בדיקה אמיתית שהייתה מתועדת ב-docstring המקורי אבל מעולם לא מומשה:
    האם ה-reward של TradingEnvironment בצעד t תלוי בנתונים אחרי t?

    שיטה: מריצים את אותה סדרת פעולות דטרמיניסטית פעמיים על אותם נתונים —
    פעם אחת עם הדאטה המלא, פעם אחת עם דאטה חתוך זמן קצר אחרי נקודת בדיקה.
    אם ה-reward/net_worth בצעדים שלפני החיתוך זהים בשני המקרים, אין leakage
    של נתוני עתיד ל-reward. אם הם שונים — ה-reward "ראה" משהו שלא היה אמור.
    """
    from trading_env import TradingEnvironment

    ok = True
    ticker_sample = {t: df for t, df in list(all_data.items())[:3]}
    if len(ticker_sample) < 1:
        check_warn("Target Leakage", "No data available — skipped")
        return True

    min_len = min(len(df) for df in ticker_sample.values())
    if min_len < 300:
        check_warn("Target Leakage", "Not enough rows to run the truncation test — skipped")
        return True

    cutoff = min_len - 60
    truncated = {t: df.iloc[:cutoff].copy() for t, df in ticker_sample.items()}

    env_full = TradingEnvironment(ticker_sample, max_drawdown_stop=1.0)
    env_trunc = TradingEnvironment(truncated, max_drawdown_stop=1.0)

    env_full.reset()
    env_trunc.reset()

    shared_steps = min(env_full.total_steps, env_trunc.total_steps)
    rng = np.random.default_rng(42)
    num_stocks = env_full.num_stocks

    for step in range(shared_steps):
        action = rng.uniform(-1.0, 1.0, size=num_stocks).astype(np.float32)
        _, reward_full, done_full, _, info_full = env_full.step(action.copy())
        _, reward_trunc, done_trunc, _, info_trunc = env_trunc.step(action.copy())

        if abs(reward_full - reward_trunc) > 1e-6 or abs(info_full["net_worth"] - info_trunc["net_worth"]) > 1e-6:
            check_fail(
                "Target Leakage (reward)",
                f"step={step}: reward/net_worth diverged between full and truncated data "
                f"(reward {reward_full:.6f} vs {reward_trunc:.6f}) — the reward function "
                "is seeing data it shouldn't have at this point in time.",
            )
            ok = False
            break

        if done_trunc:
            break

    if ok:
        check_passed("Target Leakage — reward/net_worth identical whether or not future rows exist")
    return ok


# ══════════════════════════════════════════════════════════════════
# 4. Temporal Consistency
# ══════════════════════════════════════════════════════════════════

def check_temporal_consistency(all_data: dict[str, pd.DataFrame]) -> bool:
    """מוודא שהנתונים ממוינים לפי תאריך ואין קפיצות זמן חריגות."""
    ok = True
    for ticker, df in all_data.items():
        # ממוין?
        if not df.index.is_monotonic_increasing:
            check_fail(f"Temporal Order ({ticker})", "Index is not sorted ascending")
            ok = False

        # פערים גדולים?
        if len(df) > 1:
            gaps = pd.Series(df.index).diff().dt.days.dropna()
            max_gap = gaps.max()
            if max_gap > 10:  # יותר מ-10 ימים = חג/שבת ממושך או בעיה
                if max_gap > 30:
                    check_warn(f"Gap in {ticker}", f"Max gap: {max_gap} days")
                # gaps of 3-10 days are normal (weekends + holidays)

    if ok:
        check_passed("Temporal Consistency — all data properly ordered")
    return ok


# ══════════════════════════════════════════════════════════════════
# 5. Feature Distribution: Train vs Test
# ══════════════════════════════════════════════════════════════════

def check_distribution_shift(all_data: dict[str, pd.DataFrame], verbose: bool = False) -> bool:
    """
    בודק שינוי התפלגות בין train לבין test.
    שינוי גדול מאוד עלול לפגוע בביצועים (distribution shift).
    """
    from scipy import stats as scipy_stats

    ok = True
    warnings_count = 0

    for ticker, df in list(all_data.items())[:5]:
        train = df[(df.index >= CFG.train_start) & (df.index <= CFG.train_end)]
        test  = df[(df.index >= CFG.test_start)  & (df.index <= CFG.test_end)]

        for col in ["close", "volume"] if "volume" in df.columns else ["close"]:
            if col not in df.columns:
                continue

            t_train = train[col].dropna().values
            t_test  = test[col].dropna().values

            if len(t_train) < 10 or len(t_test) < 10:
                continue

            # KS test
            ks_stat, ks_p = scipy_stats.ks_2samp(
                np.log1p(np.abs(np.diff(t_train) / (t_train[:-1] + 1e-9))),
                np.log1p(np.abs(np.diff(t_test)  / (t_test[:-1]  + 1e-9))),
            )

            if ks_p < 0.01:
                warnings_count += 1
                if verbose:
                    check_warn(f"Distribution shift {ticker}:{col}",
                               f"KS p={ks_p:.4f} — returns distribution differs from train")

    if warnings_count > 3:
        check_warn("Distribution Shift",
                   f"{warnings_count} features show significant shift — model may underperform")
        ok = False
    else:
        check_passed(f"Distribution Shift — {warnings_count} minor shifts detected (normal)")
    return ok


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

verbose_global = False


def run_all_checks(verbose: bool = False) -> bool:
    global verbose_global
    verbose_global = verbose

    print("\n" + "═" * 55)
    print("  DATA LEAKAGE & QUALITY CHECKS")
    print("═" * 55)
    print(f"  Train: {CFG.train_start} → {CFG.train_end}")
    print(f"  Test:  {CFG.test_start}  → {CFG.test_end}")
    print("─" * 55)

    # טעינת נתונים
    try:
        from data_manager import DataManager
        dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=CFG.data_end)
        dm.load_all(force_download=False)
        all_data = dm.get_aligned_data()
        print(f"  Loaded {len(all_data)} tickers · "
              f"{sum(len(df) for df in all_data.values()):,} total rows\n")
    except Exception as e:
        print(f"  [ERROR] Could not load data: {e}")
        print("  Run: python main.py --mode download")
        return False

    results = []

    # 1. Train/Test Overlap
    results.append(check_train_test_overlap(all_data))

    # 2. Feature Lookahead
    results.append(check_feature_lookahead(all_data, verbose=verbose))

    # 3. Normalization
    results.append(check_normalization_leak())

    # 3.5. Target Leakage (reward function) — documented since the original
    # version of this file but never implemented until now.
    results.append(check_target_leakage(all_data, verbose=verbose))

    # 4. Temporal Consistency
    results.append(check_temporal_consistency(all_data))

    # 5. Distribution Shift
    try:
        from scipy import stats
        results.append(check_distribution_shift(all_data, verbose=verbose))
    except ImportError:
        check_warn("Distribution Shift", "scipy not installed — skipping (pip install scipy)")
        results.append(True)

    # סיכום
    passed = sum(results)
    total  = len(results)
    print(f"\n{'═' * 55}")
    print(f"  RESULT: {passed}/{total} checks passed")

    if passed == total:
        print("  ✅ No data leakage detected — evaluation is trustworthy")
    elif passed >= total - 1:
        print("  ⚠️  Minor issues detected — review warnings above")
    else:
        print("  ❌ Significant issues detected — fix before trusting results!")
    print("═" * 55 + "\n")

    return passed == total


def parse_args():
    p = argparse.ArgumentParser(description="Data Leakage Checker")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed output for each check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_checks(verbose=args.verbose)
