"""
Master script: runs the full NBA Player Ranking pipeline in order.
Steps 1-8: Generate dataset (450 players) -> EDA (trimmed plots) -> Ranking models (updated weights)
          -> Visualize (trimmed, debiased) -> Stat leaderboards -> Position rankings (updated weights)
          -> Debiased rankings (updated weights) -> Final report.
Step 9: Validation (row counts, plots, no real NBA names, weights, ppg, outliers).
"""

import subprocess
import sys
import time
from pathlib import Path

# Project root (directory containing main.py)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

EXPECTED_PLAYERS = 450

# Distinctive real NBA player names (no common surnames like Mitchell/Donovan).
# No synthetic player name should contain any of these (word-boundary match).
REAL_NBA_NAMES = [
    "LeBron", "Curry", "Durant", "Giannis", "Antetokounmpo", "Jokic", "Doncic",
    "Tatum", "Embiid", "Westbrook", "Kawhi", "Leonard", "Harden", "Lillard",
    "Irving", "Thompson", "Draymond", "Wiggins", "Holiday", "Jrue", "Middleton",
    "Adebayo", "Zion", "Williamson", "Morant", "Trae", "Brunson", "LaMelo",
    "Cunningham", "Wembanyama", "Sengun", "Gilgeous", "Haliburton", "De'Aaron",
    "Booker", "Butler", "Sabonis", "Porzingis", "Gobert", "Vucevic", "Capela",
    "Valanciunas", "Siakam", "Anunoby", "Banchero", "Paolo", "Holmgren", "Chet", "SGA",
]
REAL_NBA_NAMES_SET = {n.strip().lower() for n in REAL_NBA_NAMES if n.strip()}

# Order: debiased_rankings must run before create_plots (plots need debiased_rankings.csv)
STEPS = [
    ("generate_dataset.py", "450 players, global names, outliers, lower scoring"),
    ("eda.py", "Data analysis, console output only (no plots)"),
    ("ranking_models.py", "5 ranking methods with corrected weights (no plots)"),
    ("stat_leaderboards.py", "Per-stat leaderboards, console output only (no plots)"),
    ("position_rankings.py", "Position-adjusted rankings with corrected weights (no plots)"),
    ("debiased_rankings.py", "Position-relative z-score rankings with corrected weights (no plots)"),
    ("create_plots.py", "ALL 23 plots generated here"),
    ("final_report.py", "Comprehensive console summary"),
]


def run_step(script_name: str, description: str) -> tuple[bool, float]:
    """
    Run a Python script in the project directory. Returns (success, elapsed_seconds).
    """
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False, 0.0
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        success = result.returncode == 0
        if not success:
            print(f"  Step failed with exit code {result.returncode}.")
        return success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print("  Step timed out after 300 seconds.")
        return False, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  Error: {e}")
        return False, elapsed


def run_validation() -> bool:
    """Run all validation checks. Returns True if all pass."""
    all_pass = True

    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)

    # 1. nba_synthetic_stats.csv has exactly 450 rows
    stats_path = DATA_DIR / "nba_synthetic_stats.csv"
    if not stats_path.exists():
        print("  [FAIL] data/nba_synthetic_stats.csv: file not found.")
        all_pass = False
    else:
        import pandas as pd
        df_stats = pd.read_csv(stats_path)
        n = len(df_stats)
        if n == EXPECTED_PLAYERS:
            print(f"  [PASS] data/nba_synthetic_stats.csv has exactly {EXPECTED_PLAYERS} rows.")
        else:
            print(f"  [FAIL] data/nba_synthetic_stats.csv has {n} rows (expected {EXPECTED_PLAYERS}).")
            all_pass = False

    # 2. final_rankings.csv has exactly 450 rows
    rank_path = DATA_DIR / "final_rankings.csv"
    if not rank_path.exists():
        print("  [FAIL] data/final_rankings.csv: file not found.")
        all_pass = False
    else:
        df_rank = pd.read_csv(rank_path)
        n = len(df_rank)
        if n == EXPECTED_PLAYERS:
            print(f"  [PASS] data/final_rankings.csv has exactly {EXPECTED_PLAYERS} rows.")
        else:
            print(f"  [FAIL] data/final_rankings.csv has {n} rows (expected {EXPECTED_PLAYERS}).")
            all_pass = False

    # 3. debiased_rankings.csv has exactly 450 rows
    debiased_path = DATA_DIR / "debiased_rankings.csv"
    if not debiased_path.exists():
        print("  [FAIL] data/debiased_rankings.csv: file not found.")
        all_pass = False
    else:
        df_deb = pd.read_csv(debiased_path)
        n = len(df_deb)
        if n == EXPECTED_PLAYERS:
            print(f"  [PASS] data/debiased_rankings.csv has exactly {EXPECTED_PLAYERS} rows.")
        else:
            print(f"  [FAIL] data/debiased_rankings.csv has {n} rows (expected {EXPECTED_PLAYERS}).")
            all_pass = False

    # 4. All 23 expected plot files exist
    EXPECTED_PLOTS = [
        "top10_overall.png", "top5_mvp.png", "top5_dpoy.png",
        "all_nba_teams.png", "all_defensive_teams.png",
        "correlation_heatmap.png", "global_vs_position_rank.png",
        "kmeans_clusters.png", "position_boxplots.png",
        "pca_biplot.png", "scatter_matrix.png",
    ] + [f"top10_{col}.png" for col in ["ppg", "rpg", "apg", "bpg", "spg", "plus_minus", 
                                         "def_rating", "off_rating", "fg_pct", "three_pt_pct", "ft_pct", "vorp"]]
    
    missing = []
    for name in EXPECTED_PLOTS:
        if not (PLOTS_DIR / name).exists():
            missing.append(name)
    if not missing:
        print(f"  [PASS] All {len(EXPECTED_PLOTS)} expected plot files exist in plots/.")
    else:
        print(f"  [FAIL] Missing plots: {missing}")
        all_pass = False

    # 5. No player name contains a known real NBA player name
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        names = df["player_name"].astype(str)
        violations = []
        for full_name in names:
            words = full_name.replace("'", " ").replace("-", " ").split()
            for w in words:
                if w.lower() in REAL_NBA_NAMES_SET:
                    violations.append((full_name, w))
                    break
        if not violations:
            print("  [PASS] No player name contains a known real NBA player name.")
        else:
            print(f"  [FAIL] Player names contain real NBA name(s): {violations[:10]}{'...' if len(violations) > 10 else ''}")
            all_pass = False

    # 6. All position-specific weights sum to 1.0
    try:
        import position_rankings as pr
        bad = []
        for pos, weights in pr.POS_WEIGHTS.items():
            s = sum(weights.values())
            if abs(s - 1.0) >= 0.001:
                bad.append(f"{pos}={s:.4f}")
        if not bad:
            print("  [PASS] All position-specific weights sum to 1.0.")
        else:
            print(f"  [FAIL] Position weights do not sum to 1.0: {bad}")
            all_pass = False
    except Exception as e:
        print(f"  [FAIL] Could not verify position weights: {e}")
        all_pass = False

    # 7. League average ppg is between 10 and 14
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        avg_ppg = df["ppg"].mean()
        if 10 <= avg_ppg <= 14:
            print(f"  [PASS] League average ppg is {avg_ppg:.2f} (expected 10-14).")
        else:
            print(f"  [FAIL] League average ppg is {avg_ppg:.2f} (expected between 10 and 14).")
            all_pass = False

    # 8. At least 25 outlier archetype players exist
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        if "archetype" not in df.columns:
            print("  [FAIL] Column 'archetype' not found in nba_synthetic_stats.csv.")
            all_pass = False
        else:
            n_outliers = df["archetype"].notna().sum()
            if n_outliers >= 25:
                print(f"  [PASS] At least 25 outlier archetype players exist ({n_outliers} found).")
            else:
                print(f"  [FAIL] Outlier archetype players: {n_outliers} (expected at least 25).")
                all_pass = False

    # 9. PPG max between 33 and 35
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        ppg_max = df["ppg"].max()
        if 33 <= ppg_max <= 35:
            print(f"  [PASS] PPG max is {ppg_max:.1f} (expected 33-35).")
        else:
            print(f"  [FAIL] PPG max is {ppg_max:.1f} (expected between 33 and 35).")
            all_pass = False

    # 10. Off Rating max between 120 and 126
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        off_max = df["off_rating"].max()
        if 120 <= off_max <= 126:
            print(f"  [PASS] Off Rating max is {off_max:.1f} (expected 120-126).")
        else:
            print(f"  [FAIL] Off Rating max is {off_max:.1f} (expected between 120 and 126).")
            all_pass = False

    # 11. FT% max at or below 0.960
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        ft_max = df["ft_pct"].max()
        if ft_max <= 0.960:
            print(f"  [PASS] FT% max is {ft_max:.3f} (expected <= 0.960).")
        else:
            print(f"  [FAIL] FT% max is {ft_max:.3f} (expected <= 0.960).")
            all_pass = False

    # 12. No FT% above 100%
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        ft_above_100 = (df["ft_pct"] > 1.0).sum()
        if ft_above_100 == 0:
            print("  [PASS] No FT% above 100%.")
        else:
            print(f"  [FAIL] {ft_above_100} player(s) have FT% > 100%.")
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("VALIDATION SUMMARY: All checks PASSED.")
    else:
        print("VALIDATION SUMMARY: One or more checks FAILED. Fix the issues above.")
    print("=" * 50)
    return all_pass


def main():
    print("NBA Player Ranking Engine - Full Pipeline")
    print("=" * 50)
    total_start = time.perf_counter()
    results = []

    for i, (script, description) in enumerate(STEPS, start=1):
        print(f"\n=== Step {i}/{len(STEPS)}: {description}... ===")
        success, elapsed = run_step(script, description)
        results.append((script, success, elapsed))
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {elapsed:.1f}s")

    # Validation (only if all steps succeeded so files exist)
    validation_ok = True
    if all(s for _, s, _ in results):
        validation_ok = run_validation()
    else:
        print("\n  Skipping validation (one or more pipeline steps failed).")

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 50)
    print("Pipeline summary")
    print("=" * 50)
    for script, success, elapsed in results:
        status = "OK" if success else "FAILED"
        print(f"  {script}: {status} ({elapsed:.1f}s)")
    print(f"  Total runtime: {total_elapsed:.1f}s")
    if all(s for _, s, _ in results):
        print(f"  Validation: {'PASSED' if validation_ok else 'FAILED'}")
    failed = sum(1 for _, s, _ in results if not s)
    if failed:
        print(f"\n  {failed} step(s) failed. Check output above.")
        sys.exit(1)
    if not validation_ok:
        print("\n  Validation failed. Fix the reported issues.")
        sys.exit(1)
    print("\n  All steps and validation completed successfully.")


if __name__ == "__main__":
    main()
