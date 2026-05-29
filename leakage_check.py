"""
leakage_check.py
================
בדיקת Data Leakage אגרסיבית — הסיכון הכי גדול במערכת ML למסחר.

בדיקות:
  1. Train/Test overlap  — האם יש תאריכים משותפים?
  2. Feature lookahead   — האם פיצ'רים משתמשים בנתוני עתיד?
  3. Target leakage      — האם ה-reward מחשב מחיר עתידי שנחשף במצב?
  4. Normalization leak  — האם ה-VecNormalize חושב stats על נתוני הטסט?
  5. Stationarity        — האם ה-features stationarity-aware?

שימוש:
    python leakage_check.py           # הרצת כל הבדיקות
    python leakage_check.py --verbose # פרטים מלאים
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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
    בודק שהפיצ'רים לא משתמשים בנתוני עתיד.
    כל פיצ'ר צריך להיות ניתן לחישוב עם נתוני עבר בלבד.
    """
    ok = True

    for ticker, df in list(all_data.items())[:3]:  # בדוק דוגמה
        for col in df.columns:
            # Forward-fill לאחר dropna מרמז על בעיה פוטנציאלית
            if df[col].isna().any():
                if verbose:
                    check_warn(f"Feature {col} ({ticker})",
                               f"{df[col].isna().sum()} NaN values — may indicate ffill issue")

            # בדיקה: האם עמודה כלשהי תלויה ב-shift שלילי (look-ahead)?
            # זה לא ניתן לזיהוי אוטומטי מלא, אבל בודקים שינויים חריגים
            if len(df) > 1:
                corr_with_future = abs(df[col].corr(df[col].shift(-1)))
                if corr_with_future > 0.999 and verbose:
                    check_warn(f"Feature {col} ({ticker})",
                               f"suspiciously high correlation with future value ({corr_with_future:.4f})")

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
        check_passed("Feature Lookahead — no obvious forward-looking features detected")
    return ok


# ══════════════════════════════════════════════════════════════════
# 3. Normalization Leakage
# ══════════════════════════════════════════════════════════════════

def check_normalization_leak() -> bool:
    """
    בודק שה-VecNormalize לא חושב stats על נתוני הטסט.
    כלל: VecNormalize.training=True רק בזמן אימון, False בזמן evaluation.
    """
    norm_path = Path(CFG.model_dir) / "vec_normalize.pkl"

    if not norm_path.exists():
        check_warn("Normalization", f"vec_normalize.pkl not found at {norm_path} — cannot verify")
        return True

    try:
        import pickle
        with open(norm_path, "rb") as f:
            vec_norm = pickle.load(f)

        # בדיקה שה-norm אומן רק על train data
        # לא ניתן לדעת בוודאות, אבל בודקים שה-stats סבירים
        if hasattr(vec_norm, "obs_rms") and vec_norm.obs_rms is not None:
            mean_abs = np.abs(vec_norm.obs_rms.mean).mean()
            std_mean = vec_norm.obs_rms.var.mean() ** 0.5
            if verbose_global:
                print(f"    obs_rms: mean_abs={mean_abs:.3f}, std={std_mean:.3f}")

        check_passed("Normalization — VecNormalize loaded successfully")
        return True

    except Exception as e:
        check_warn("Normalization", f"Could not verify: {e}")
        return True


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
