"""
Position-adjusted rankings and analyses.

- Loads:
  - data/nba_synthetic_stats.csv
  - data/final_rankings.csv

Produces:
- Per-position weighted composite rankings (PG/SG/SF/PF/C) with position-specific weights.
- Position stat leaders (ppg, rpg, apg, and a key stat per position).
- All-NBA First/Second/Third Teams based on position-adjusted ranks.
- Comparison of within-position rank vs global consensus rank, incl. overrated/underrated.
- Note: All plot generation has been moved to create_plots.py
"""

import os
import numpy as np
import pandas as pd

STATS_PATH = "data/nba_synthetic_stats.csv"
RANKINGS_PATH = "data/final_rankings.csv"

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

# Position-specific weights (must sum to 1.0 per position)
POS_WEIGHTS = {
    "PG": {
        "ppg": 0.20,
        "apg": 0.20,
        "vorp": 0.13,
        "plus_minus": 0.13,
        "spg": 0.07,
        "off_rating": 0.07,
        "three_pt_pct": 0.05,
        "def_rating": 0.04,
        "fg_pct": 0.04,
        "ft_pct": 0.02,
        "rpg": 0.03,
        "bpg": 0.02,
    },
    "SG": {
        "ppg": 0.23,
        "vorp": 0.14,
        "plus_minus": 0.12,
        "apg": 0.10,
        "three_pt_pct": 0.06,
        "off_rating": 0.07,
        "spg": 0.06,
        "fg_pct": 0.06,
        "def_rating": 0.05,
        "ft_pct": 0.03,
        "rpg": 0.06,
        "bpg": 0.02,
    },
    "SF": {
        "ppg": 0.19,
        "vorp": 0.16,
        "plus_minus": 0.10,
        "apg": 0.10,
        "def_rating": 0.09,
        "off_rating": 0.08,
        "rpg": 0.10,
        "spg": 0.05,
        "bpg": 0.04,
        "fg_pct": 0.04,
        "three_pt_pct": 0.04,
        "ft_pct": 0.01,
    },
    "PF": {
        "ppg": 0.20,
        "vorp": 0.16,
        "plus_minus": 0.12,
        "rpg": 0.10,
        "def_rating": 0.09,
        "bpg": 0.07,
        "apg": 0.06,
        "off_rating": 0.06,
        "fg_pct": 0.05,
        "spg": 0.05,
        "three_pt_pct": 0.03,
        "ft_pct": 0.01,
    },
    "C": {
        "ppg": 0.18,
        "vorp": 0.16,
        "plus_minus": 0.12,
        "rpg": 0.12,
        "bpg": 0.10,
        "def_rating": 0.10,
        "fg_pct": 0.07,
        "off_rating": 0.05,
        "apg": 0.04,
        "spg": 0.03,
        "three_pt_pct": 0.01,
        "ft_pct": 0.02,
    },
}
# Verify all position weights sum to 1.0
for pos, weights in POS_WEIGHTS.items():
    assert abs(sum(weights.values()) - 1.0) < 0.001, f"{pos} weights sum to {sum(weights.values())}, not 1.0"

# Print confirmation of weight updates (when script runs directly)
if __name__ == "__main__":
    print("position_rankings.py: Position weights updated")
    print("  PG: vorp 0.10->0.13 (+0.03), apg 0.18->0.20 (+0.02), plus_minus 0.12->0.13 (+0.01)")
    print("  SG: vorp 0.10->0.14 (+0.04), apg 0.08->0.10 (+0.02), ppg 0.22->0.23 (+0.01)")
    print("  SF: vorp 0.13->0.16 (+0.03), apg 0.07->0.10 (+0.03), rpg 0.08->0.10 (+0.02)")
    print("  PF: vorp 0.13->0.16 (+0.03), ppg 0.16->0.20 (+0.04), apg 0.05->0.06 (+0.01)")
    print("  C: vorp 0.14->0.16 (+0.02), ppg 0.14->0.18 (+0.04), rpg 0.14->0.12 (-0.02)")

# Key stat per position for "position stat leaders"
POSITION_KEY_STAT = {
    "PG": "apg",
    "SG": "three_pt_pct",
    "SF": "vorp",
    "PF": "rpg",
    "C": "bpg",
}

STAT_LABELS = {
    "ppg": "PPG",
    "rpg": "RPG",
    "apg": "APG",
    "bpg": "BPG",
    "spg": "SPG",
    "plus_minus": "+/-",
    "def_rating": "Def Rating",
    "off_rating": "Off Rating",
    "fg_pct": "FG%",
    "three_pt_pct": "3P%",
    "ft_pct": "FT%",
    "vorp": "VORP",
}



def load_merged() -> pd.DataFrame:
    stats = pd.read_csv(STATS_PATH)
    rankings = pd.read_csv(RANKINGS_PATH)
    merged = rankings.merge(stats, on=["player_name", "position"], how="left")
    return merged


def zscore_within_position(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize STAT_COLS within a position group.
    def_rating is flipped so higher is better before normalization.
    """
    X = df_pos[STAT_COLS].copy()
    X["def_rating"] = -X["def_rating"]
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = stds.replace(0, 1)
    Z = (X - means) / stds
    return Z


def compute_position_scores(merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each position, compute position-adjusted composite score and rank.
    Adds columns:
      - position_score
      - position_rank
    """
    merged = merged.copy()
    merged["position_score"] = np.nan
    merged["position_rank"] = np.nan

    for pos, weights in POS_WEIGHTS.items():
        mask = merged["position"] == pos
        df_pos = merged.loc[mask].copy()
        if df_pos.empty:
            continue
        Z = zscore_within_position(df_pos)
        w = pd.Series(weights)
        # Ensure all STAT_COLS have a weight (default 0)
        for col in STAT_COLS:
            if col not in w:
                w[col] = 0.0
        w = w[STAT_COLS]
        score = (Z * w).sum(axis=1)
        # Rank within position (1 = best)
        rank = score.rank(ascending=False, method="min")
        merged.loc[mask, "position_score"] = score
        merged.loc[mask, "position_rank"] = rank

    merged["position_rank"] = merged["position_rank"].astype(int)
    return merged


def print_position_stat_leaders(merged: pd.DataFrame):
    print("=" * 80)
    print("PER-POSITION STAT LEADERS")
    print("=" * 80)
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        df_pos = merged[merged["position"] == pos]
        if df_pos.empty:
            continue
        key_stat = POSITION_KEY_STAT[pos]
        stats_of_interest = ["ppg", "rpg", "apg", key_stat]
        labels = [STAT_LABELS[s] for s in stats_of_interest]

        print(f"\n--- {pos} Leaders ---")
        for s, label in zip(stats_of_interest, labels):
            top5 = df_pos.nlargest(5, s)[["player_name", "position", s]].copy()
            print(f"\nTop 5 {pos} in {label}:")
            rows = []
            for rank, (_, row) in enumerate(top5.iterrows(), start=1):
                val = row[s]
                if s.endswith("_pct"):
                    vstr = f"{val:.3f}"
                else:
                    vstr = f"{val:.1f}"
                rows.append(
                    {
                        "Rank": rank,
                        "Player": row["player_name"],
                        "Pos": row["position"],
                        label: vstr,
                    }
                )
            print(pd.DataFrame(rows).to_string(index=False))


def build_all_nba_teams(merged: pd.DataFrame):
    """
    Returns dict: team_name -> list of player rows (PG, SG, SF, PF, C)
    Based on position_rank (best = rank 1, etc.)
    """
    teams = {"First Team": [], "Second Team": [], "Third Team": []}
    order = ["PG", "SG", "SF", "PF", "C"]
    for team_idx, team_name in enumerate(teams.keys(), start=1):
        for pos in order:
            df_pos = merged[merged["position"] == pos].sort_values(
                "position_rank"
            )
            if len(df_pos) >= team_idx:
                teams[team_name].append(df_pos.iloc[team_idx - 1])
    return teams


def print_all_nba_teams(teams):
    print("\n" + "=" * 80)
    print("ALL-NBA TEAMS (Position-Adjusted)")
    print("=" * 80)
    for team_name, players in teams.items():
        print(f"\n{team_name}:")
        rows = []
        for row in players:
            rows.append(
                {
                    "Pos": row["position"],
                    "Player": row["player_name"],
                    "PPG": f"{row['ppg']:.1f}",
                    "RPG": f"{row['rpg']:.1f}",
                    "APG": f"{row['apg']:.1f}",
                    "VORP": f"{row['vorp']:.1f}",
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))


def comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    comp = merged[
        ["player_name", "position", "consensus_rank", "position_rank"]
    ].copy()
    comp["diff"] = comp["consensus_rank"] - comp["position_rank"]
    # diff > 0: consensus rank is worse (higher number) than position rank -> UNDERRATED
    # diff < 0: consensus rank is better (lower number) than position rank -> OVERRATED
    comp_sorted = comp.sort_values("consensus_rank")
    return comp_sorted


def print_overrated_underrated(comp: pd.DataFrame):
    print("\n" + "=" * 80)
    print("GLOBAL VS POSITION RANK DISCREPANCIES")
    print("=" * 80)
    # Most overrated: global rank much better than position rank (diff << 0)
    over = comp.sort_values("diff").head(5)
    under = comp.sort_values("diff", ascending=False).head(5)

    def fmt_row(row):
        return (
            f"{row['player_name']} ({row['position']}): "
            f"global #{row['consensus_rank']} vs position #{row['position_rank']} "
            f"(diff={row['diff']:+d})"
        )

    print("\nMOST OVERRATED BY GLOBAL RANKINGS (global rank better than position rank):")
    for _, r in over.iterrows():
        print("  - " + fmt_row(r))

    print("\nMOST UNDERRATED BY GLOBAL RANKINGS (position rank better than global rank):")
    for _, r in under.iterrows():
        print("  - " + fmt_row(r))


def main():
    merged = load_merged()
    merged = compute_position_scores(merged)

    print_position_stat_leaders(merged)
    teams = build_all_nba_teams(merged)
    print_all_nba_teams(teams)

    comp = comparison_table(merged)
    print("\n" + "=" * 80)
    print("GLOBAL VS POSITION RANK TABLE (sorted by global consensus rank)")
    print("=" * 80)
    print(
        comp.to_string(
            index=False,
            columns=["player_name", "position", "consensus_rank", "position_rank", "diff"],
        )
    )
    print_overrated_underrated(comp)

    print("\nNote: All plot generation has been moved to create_plots.py")


if __name__ == "__main__":
    main()

