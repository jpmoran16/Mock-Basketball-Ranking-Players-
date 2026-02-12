"""
Generate detailed stat leaderboards from nba_synthetic_stats.csv.

For each stat, prints a top-10 table with:
- Rank
- Player
- Position
- Stat value
- % above/below league average

Also:
- Counts top-10 appearances per player (3+ = well-rounded).
- Prints #1 leader for each stat.
- Flags positional surprise outliers.
- Note: All plot generation has been moved to create_plots.py
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = "data/nba_synthetic_stats.csv"

STAT_COLS = [
    "ppg",
    "rpg",
    "apg",
    "bpg",
    "spg",
    "plus_minus",
    "off_rating",
    "def_rating",  # lower is better
    "fg_pct",
    "three_pt_pct",
    "ft_pct",
    "vorp",
]

STAT_LABELS = {
    "ppg": "PPG",
    "rpg": "RPG",
    "apg": "APG",
    "bpg": "BPG",
    "spg": "SPG",
    "plus_minus": "+/-",
    "off_rating": "Off Rating",
    "def_rating": "Def Rating",
    "fg_pct": "FG%",
    "three_pt_pct": "3P%",
    "ft_pct": "FT%",
    "vorp": "VORP",
}



def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def compute_leaderboards(df: pd.DataFrame):
    """
    Compute top-10 leaderboards for each stat.
    Returns:
      leaderboards: dict[stat] -> DataFrame with columns:
        rank, player_name, position, value, pct_above_avg
      appearances: dict[player_name] -> {"position": pos, "stats": set of stat keys}
    """
    leaderboards = {}
    appearances = {}

    for stat in STAT_COLS:
        col = df[stat]
        mean_val = col.mean()
        if stat == "def_rating":
            # Lower is better; "above avg" as how much lower than avg
            ascending = True
        else:
            ascending = False

        top = df.nsmallest(10, stat) if ascending else df.nlargest(10, stat)
        top = top.reset_index(drop=True)
        ranks = np.arange(1, len(top) + 1)

        values = top[stat].astype(float)
        if mean_val != 0:
            if stat == "def_rating":
                pct = (mean_val - values) / mean_val * 100.0
            else:
                pct = (values - mean_val) / mean_val * 100.0
        else:
            pct = pd.Series([0.0] * len(values), index=values.index)

        lb = pd.DataFrame(
            {
                "rank": ranks,
                "player_name": top["player_name"],
                "position": top["position"],
                "value": values,
                "pct_above_avg": pct,
                "league_avg": mean_val,
            }
        )
        leaderboards[stat] = lb

        # Update appearances
        for _, row in lb.iterrows():
            name = row["player_name"]
            pos = row["position"]
            if name not in appearances:
                appearances[name] = {"position": pos, "stats": set()}
            appearances[name]["stats"].add(stat)

    return leaderboards, appearances


def format_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def print_leaderboards(leaderboards):
    print("=" * 80)
    print("INDIVIDUAL STAT LEADERBOARDS (Top 10)")
    print("=" * 80)

    for stat in STAT_COLS:
        lb = leaderboards[stat]
        label = STAT_LABELS[stat]
        print(f"\n--- {label} Top 10 ---")
        rows = []
        for _, row in lb.iterrows():
            rows.append(
                {
                    "Rank": int(row["rank"]),
                    "Player": row["player_name"],
                    "Pos": row["position"],
                    label: row["value"],
                    "% vs Avg": format_pct(row["pct_above_avg"]),
                }
            )
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))


def print_most_appearances(appearances):
    rows = []
    for name, info in appearances.items():
        count = len(info["stats"])
        if count >= 3:
            rows.append(
                {
                    "Player": name,
                    "Position": info["position"],
                    "Top10_Count": count,
                    "Stats": ", ".join(sorted(STAT_LABELS[s] for s in info["stats"])),
                }
            )
    if not rows:
        print("\nNo players appear on 3 or more leaderboards.")
        return

    df = pd.DataFrame(rows).sort_values(
        ["Top10_Count", "Player"], ascending=[False, True]
    )
    print("\n" + "=" * 80)
    print('"MOST TOP-10 APPEARANCES" (players on 3+ leaderboards)')
    print("=" * 80)
    print(df.to_string(index=False))


def print_stat_kings(leaderboards):
    print("\n" + "=" * 80)
    print('"STAT KINGS" (stat leaders)')
    print("=" * 80)
    for stat in STAT_COLS:
        lb = leaderboards[stat]
        top = lb.iloc[0]
        label = STAT_LABELS[stat]
        val = top["value"]
        if stat.endswith("_pct"):
            val_str = f"{val:.3f}"
        else:
            val_str = f"{val:.1f}"
        print(
            f"{label} Leader: {top['player_name']} ({top['position']}) - {val_str}"
        )


def print_positional_surprises(leaderboards):
    print("\n" + "=" * 80)
    print("POSITIONAL SURPRISES (OUTLIER ALERTS)")
    print("=" * 80)
    alerts = []

    for stat in STAT_COLS:
        lb = leaderboards[stat]
        for _, row in lb.iterrows():
            pos = row["position"]
            rank = int(row["rank"])
            val = row["value"]
            name = row["player_name"]
            label = STAT_LABELS[stat]

            # Guards in rpg or bpg
            if pos in {"PG", "SG"} and stat in {"rpg", "bpg"}:
                alerts.append(
                    f'OUTLIER ALERT: {name} ({pos}) ranks #{rank} in {label} with {val:.1f}'
                )
            # Centers in apg, spg, three_pt_pct, ft_pct
            if pos == "C" and stat in {"apg", "spg", "three_pt_pct", "ft_pct"}:
                if stat.endswith("_pct"):
                    vstr = f"{val:.3f}"
                else:
                    vstr = f"{val:.1f}"
                alerts.append(
                    f'OUTLIER ALERT: {name} ({pos}) ranks #{rank} in {label} with {vstr}'
                )
            # Forwards in apg
            if pos in {"SF", "PF"} and stat == "apg":
                alerts.append(
                    f'OUTLIER ALERT: {name} ({pos}) ranks #{rank} in {label} with {val:.1f}'
                )

    if not alerts:
        print("No positional surprises found.")
    else:
        for line in alerts:
            print(line)


def main():
    df = load_data()
    leaderboards, appearances = compute_leaderboards(df)

    print_leaderboards(leaderboards)
    print_most_appearances(appearances)
    print_stat_kings(leaderboards)
    print_positional_surprises(leaderboards)

    print("\nNote: All plot generation has been moved to create_plots.py")


if __name__ == "__main__":
    main()

