"""
Debiased global player rankings using position-relative z-scores.

Idea:
- Instead of z-scoring each stat vs the whole league, z-score within POSITION.
- Flip def_rating so higher is better.
- Apply the same expert weights as Method 1 in ranking_models.py.

Outputs:
- data/debiased_rankings.csv with new debiased ranks.
- Console analysis of biggest risers/fallers and position-level effects.
- Mann-Whitney U tests to quantify guard vs big bias (old vs new ranks).
- Note: All plot generation has been moved to create_plots.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

STATS_PATH = "data/nba_synthetic_stats.csv"
RANKINGS_PATH = "data/final_rankings.csv"
OUT_CSV_PATH = "data/debiased_rankings.csv"

STAT_COLS = [
    "ppg",
    "rpg",
    "apg",
    "bpg",
    "spg",
    "plus_minus",
    "def_rating",
    "off_rating",
    "fg_pct",
    "three_pt_pct",
    "ft_pct",
    "vorp",
]

# Same expert weights as Method 1 in ranking_models.py
EXPERT_WEIGHTS = {
    "ppg": 0.21,
    "rpg": 0.06,
    "apg": 0.13,
    "bpg": 0.05,
    "spg": 0.05,
    "plus_minus": 0.10,
    "def_rating": 0.07,  # flipped so higher is better
    "off_rating": 0.07,
    "fg_pct": 0.06,
    "three_pt_pct": 0.03,
    "ft_pct": 0.01,
    "vorp": 0.16,
}
# Verify weights sum to 1.0
assert abs(sum(EXPERT_WEIGHTS.values()) - 1.0) < 0.001, f"Weights sum to {sum(EXPERT_WEIGHTS.values())}, not 1.0"

# Print confirmation of weight updates (when script runs directly)
if __name__ == "__main__":
    print("debiased_rankings.py: Global weights updated")
    print("  Top 3 changes: vorp 0.11->0.16 (+0.05), apg 0.10->0.13 (+0.03), ppg 0.20->0.21 (+0.01)")



def load_merged() -> pd.DataFrame:
    stats = pd.read_csv(STATS_PATH)
    rankings = pd.read_csv(RANKINGS_PATH)
    merged = rankings.merge(stats, on=["player_name", "position"], how="left")
    return merged


def compute_position_relative_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stat, compute z-scores within each position group.
    def_rating is flipped after z-scoring (so higher z = better defense).
    Returns a DataFrame Z with same index and STAT_COLS.
    """
    Z = pd.DataFrame(index=df.index, columns=STAT_COLS, dtype=float)
    for col in STAT_COLS:
        grp = df.groupby("position")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, 1)
        z = (df[col] - mean) / std
        if col == "def_rating":
            z = -z
        Z[col] = z
    return Z


def compute_debiased_scores(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    Z = compute_position_relative_zscores(merged)
    w = pd.Series([EXPERT_WEIGHTS[c] for c in STAT_COLS], index=STAT_COLS)
    score = (Z * w).sum(axis=1)
    merged["debiased_score"] = score
    merged["debiased_rank"] = score.rank(ascending=False, method="min").astype(int)
    return merged


def comparison_df(merged: pd.DataFrame) -> pd.DataFrame:
    comp = merged[
        ["player_name", "position", "consensus_rank", "debiased_rank"]
    ].copy()
    comp["rank_change"] = comp["consensus_rank"] - comp["debiased_rank"]
    # Positive rank_change => player moved UP (better) in debiased rankings
    return comp.sort_values("consensus_rank")


def print_risers_and_fallers(comp: pd.DataFrame):
    print("=" * 80)
    print("DEBIASED vs ORIGINAL RANKINGS")
    print("=" * 80)

    # Biggest risers: largest positive change
    risers = comp.sort_values("rank_change", ascending=False).head(10)
    fallers = comp.sort_values("rank_change", ascending=True).head(10)

    def fmt_row(row):
        return (
            f"{row['player_name']} ({row['position']}): "
            f"old #{row['consensus_rank']}, new #{row['debiased_rank']} "
            f"(change={row['rank_change']:+d})"
        )

    print("\nBIGGEST RISERS (improved the most in debiased rankings):")
    for _, r in risers.iterrows():
        print("  - " + fmt_row(r))

    print("\nBIGGEST FALLERS (dropped the most in debiased rankings):")
    for _, r in fallers.iterrows():
        print("  - " + fmt_row(r))

    # Average rank change by position
    print("\nAVERAGE RANK CHANGE BY POSITION (old_rank - new_rank):")
    by_pos = (
        comp.groupby("position")["rank_change"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(by_pos.to_string())

    # New Top 25 leaderboard
    print("\nNEW TOP 25 OVERALL (DEBIASED RANK):")
    top25 = (
        comp.sort_values("debiased_rank")
        .head(25)
        .loc[:, ["player_name", "position", "consensus_rank", "debiased_rank", "rank_change"]]
    )
    print(
        top25.rename(
            columns={
                "player_name": "Player",
                "position": "Pos",
                "consensus_rank": "OldRank",
                "debiased_rank": "NewRank",
                "rank_change": "Change",
            }
        ).to_string(index=False)
    )


def mann_whitney_tests(comp: pd.DataFrame):
    """
    Compare old vs new ranks for guards (PG+SG) vs bigs (PF+C) using Mann–Whitney U.
    """
    guards_mask = comp["position"].isin(["PG", "SG"])
    bigs_mask = comp["position"].isin(["PF", "C"])

    old_guards = comp.loc[guards_mask, "consensus_rank"].values
    old_bigs = comp.loc[bigs_mask, "consensus_rank"].values
    new_guards = comp.loc[guards_mask, "debiased_rank"].values
    new_bigs = comp.loc[bigs_mask, "debiased_rank"].values

    print("\n" + "=" * 80)
    print("MANN–WHITNEY U TEST: GUARDS vs BIGS (OLD vs NEW RANKS)")
    print("=" * 80)

    u_old, p_old = mannwhitneyu(old_guards, old_bigs, alternative="two-sided")
    u_new, p_new = mannwhitneyu(new_guards, new_bigs, alternative="two-sided")

    print(f"Old ranks:  U = {u_old:.1f}, p = {p_old:.4f}")
    print(f"New ranks:  U = {u_new:.1f}, p = {p_new:.4f}")

    def interp(p):
        if p < 0.001:
            return "strong evidence of bias (p < 0.001)"
        if p < 0.05:
            return "statistically significant difference (p < 0.05)"
        return "no statistically significant difference (p >= 0.05)"

    print(f"Interpretation (old): {interp(p_old)}")
    print(f"Interpretation (new): {interp(p_new)}")


def main():
    merged = load_merged()
    merged = compute_debiased_scores(merged)
    comp = comparison_df(merged)

    print_risers_and_fallers(comp)
    mann_whitney_tests(comp)

    # Save full debiased rankings
    merged.to_csv(OUT_CSV_PATH, index=False)
    print(f"\nDebiased rankings saved to {OUT_CSV_PATH}")

    print("\nNote: All plot generation has been moved to create_plots.py")


if __name__ == "__main__":
    main()

