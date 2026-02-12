"""
Final comprehensive report: loads all CSV outputs and prints a formatted console report.
Run after the full pipeline (generate_dataset → ... → debiased_rankings).
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from position_rankings import (
    STAT_COLS,
    POS_WEIGHTS,
    compute_position_scores,
    build_all_nba_teams,
)

STATS_PATH = "data/nba_synthetic_stats.csv"
RANKINGS_PATH = "data/final_rankings.csv"
DEBIASED_PATH = "data/debiased_rankings.csv"

STAT_LABELS = {
    "ppg": "PPG", "rpg": "RPG", "apg": "APG", "bpg": "BPG", "spg": "SPG",
    "plus_minus": "+/-", "def_rating": "DefRtg", "off_rating": "OffRtg",
    "fg_pct": "FG%", "three_pt_pct": "3P%", "ft_pct": "FT%", "vorp": "VORP",
}


def load_data():
    stats = pd.read_csv(STATS_PATH)
    rankings = pd.read_csv(RANKINGS_PATH)
    debiased = pd.read_csv(DEBIASED_PATH)
    return stats, rankings, debiased


def section_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def sub_header(title: str):
    print(f"\n--- {title} ---")


def league_summary(stats: pd.DataFrame):
    section_header("1. LEAGUE SUMMARY")
    n = len(stats)
    print(f"\nTotal players: {n}")
    print("\nPlayers per position:")
    pos_counts = stats["position"].value_counts().reindex(["PG", "SG", "SF", "PF", "C"])
    for pos, count in pos_counts.items():
        print(f"  {pos}: {count}")
    print("\nLeague average (numeric stats):")
    numeric = stats.select_dtypes(include=[np.number])
    for col in STAT_COLS:
        if col not in numeric.columns:
            continue
        avg = numeric[col].mean()
        if col.endswith("_pct"):
            print(f"  {STAT_LABELS.get(col, col)}: {avg:.3f}")
        else:
            print(f"  {STAT_LABELS.get(col, col)}: {avg:.2f}")
    sub_header("Scoring distribution (PPG)")
    ppg = stats["ppg"]
    bins = [0, 10, 15, 20, 25, 100]
    labels = ["<10", "10-15", "15-20", "20-25", "25+"]
    dist = pd.cut(ppg, bins=bins, labels=labels).value_counts().sort_index()
    for label in labels:
        pct = 100 * dist.get(label, 0) / n
        print(f"  {label}: {pct:.1f}%")


def stat_kings(stats: pd.DataFrame):
    section_header("2. STAT KINGS")
    for col in STAT_COLS:
        if col not in stats.columns:
            continue
        if col == "def_rating":
            row = stats.loc[stats[col].idxmin()]
            val = row[col]
        else:
            row = stats.loc[stats[col].idxmax()]
            val = row[col]
        if col.endswith("_pct"):
            val_str = f"{val:.3f}"
        else:
            val_str = f"{val:.1f}"
        print(f"  {STAT_LABELS.get(col, col)}: {row['player_name']} ({row['position']}) — {val_str}")


def outlier_spotlight(stats: pd.DataFrame):
    section_header("3. OUTLIER SPOTLIGHT (Unicorn / Archetype Players)")
    if "archetype" not in stats.columns:
        print("\n  No archetype column found.")
        return
    out = stats[stats["archetype"].notna()].copy()
    if out.empty:
        print("\n  No outlier archetypes in dataset.")
        return
    reasons = {
        "Point Center": "Center with elite passing (7+ APG), high RPG, finesse (lower BPG).",
        "Rebounding Guard": "Guard with 8+ RPG and solid APG (Westbrook/Luka style).",
        "Shot-blocking Wing": "SF/SG with 2.5+ BPG and strong RPG.",
        "Scoring Big with Range": "C/PF with 38%+ 3P and 20+ PPG, stretch big.",
        "Defensive Guard Menace": "PG/SG with 2.2+ SPG and elite DefRtg (96-101).",
        "Empty Stats Guy": "High PPG but bad +/- and DefRtg, low FG%, negative VORP.",
        "Quiet MVP": "Elite +/- and DefRtg, high VORP and FG%, not highest PPG.",
    }
    for _, row in out.iterrows():
        arch = row["archetype"]
        reason = reasons.get(arch, "Unusual stat profile for position.")
        print(f"\n  {row['player_name']} ({row['position']}) — {arch}")
        print(f"    Why: {reason}")


def top25_debiased(debiased: pd.DataFrame):
    section_header("4. TOP 25 OVERALL (DEBIASED — AUTHORITATIVE RANKING)")
    top = debiased.nsmallest(25, "debiased_rank")
    cols = ["debiased_rank", "player_name", "position", "ppg", "rpg", "apg", "vorp", "debiased_score"]
    df = top[cols].copy()
    df["debiased_rank"] = df["debiased_rank"].astype(int)
    df = df.rename(columns={
        "debiased_rank": "Rank",
        "player_name": "Player",
        "position": "Pos",
        "debiased_score": "Score",
    })
    for c in ["ppg", "rpg", "apg", "vorp"]:
        df[c] = df[c].round(1)
    df["Score"] = df["Score"].round(3)
    print("\n" + df.to_string(index=False))


def all_nba_teams(debiased: pd.DataFrame):
    section_header("5. ALL-NBA TEAMS (FROM POSITION RANKINGS)")
    merged = debiased.copy()
    if "position_rank" not in merged.columns or merged["position_rank"].isna().all():
        merged = compute_position_scores(merged)
    teams = build_all_nba_teams(merged)
    for team_name, players in teams.items():
        sub_header(team_name)
        rows = []
        for row in players:
            rows.append({
                "Pos": row["position"],
                "Player": row["player_name"],
                "PPG": f"{row['ppg']:.1f}",
                "RPG": f"{row['rpg']:.1f}",
                "APG": f"{row['apg']:.1f}",
                "VORP": f"{row['vorp']:.1f}",
            })
        print(pd.DataFrame(rows).to_string(index=False))


def most_well_rounded(stats: pd.DataFrame):
    section_header("6. MOST WELL-ROUNDED (TOP 5 BY TOP-10 STAT APPEARANCES)")
    appearances = {}
    for col in STAT_COLS:
        if col not in stats.columns:
            continue
        if col == "def_rating":
            top10 = stats.nsmallest(10, col)["player_name"].tolist()
        else:
            top10 = stats.nlargest(10, col)["player_name"].tolist()
        for name in top10:
            appearances[name] = appearances.get(name, 0) + 1
    sorted_ = sorted(appearances.items(), key=lambda x: -x[1])[:5]
    rows = [{"Player": name, "Top10_Count": c} for name, c in sorted_]
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def position_rankings_summary(debiased: pd.DataFrame):
    section_header("7. POSITION RANKINGS SUMMARY (TOP 5 PER POSITION)")
    merged = debiased.copy()
    if "position_rank" not in merged.columns or merged["position_rank"].isna().all():
        merged = compute_position_scores(merged)
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        sub_header(pos)
        df_pos = merged[merged["position"] == pos].sort_values("position_rank").head(5)
        out = df_pos[["position_rank", "player_name", "position_score"]].copy()
        out = out.rename(columns={"position_rank": "Rank", "player_name": "Player", "position_score": "Score"})
        out["Score"] = out["Score"].round(3)
        print(out.to_string(index=False))


def method_agreement(rankings: pd.DataFrame, debiased: pd.DataFrame):
    section_header("8. METHOD AGREEMENT SUMMARY")
    merge_cols = ["player_name", "position"]
    rank_cols = ["rank_weighted", "rank_pca", "rank_percentile", "rank_bayesian"]
    df = rankings[merge_cols + rank_cols].merge(
        debiased[merge_cols + ["debiased_rank"]],
        on=merge_cols,
        how="inner",
    )
    df = df.rename(columns={"debiased_rank": "rank_debiased"})
    methods = rank_cols + ["rank_debiased"]
    labels = ["Weighted", "PCA", "Percentile", "Bayesian", "Debiased"]
    corr = df[methods].corr(method="spearman")
    corr.columns = labels
    corr.index = labels
    print("\nSpearman correlation matrix (ranking methods):")
    print(corr.round(3).to_string())
    # Highest and lowest off-diagonal pairs
    vals = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vals.append((labels[i], labels[j], corr.iloc[i, j]))
    vals.sort(key=lambda x: -x[2])
    print("\n  Highest agreement pair:", f"{vals[0][0]} vs {vals[0][1]}: r = {vals[0][2]:.3f}")
    print("  Lowest agreement pair:", f"{vals[-1][0]} vs {vals[-1][1]}: r = {vals[-1][2]:.3f}")


def bias_report(debiased: pd.DataFrame):
    section_header("9. BIAS REPORT (DEBIASED ANALYSIS)")
    if "consensus_rank" not in debiased.columns or "debiased_rank" not in debiased.columns:
        print("\n  Missing consensus_rank or debiased_rank.")
        return
    debiased = debiased.copy()
    debiased["rank_change"] = debiased["consensus_rank"] - debiased["debiased_rank"]
    sub_header("Average rank change by position (positive = moved up in debiased)")
    by_pos = debiased.groupby("position")["rank_change"].agg(["mean", "median", "count"]).round(2)
    print(by_pos.to_string())
    sub_header("Mann-Whitney U: Guards (PG+SG) vs Bigs (PF+C)")
    guards = debiased[debiased["position"].isin(["PG", "SG"])]
    bigs = debiased[debiased["position"].isin(["PF", "C"])]
    u_old, p_old = mannwhitneyu(guards["consensus_rank"], bigs["consensus_rank"], alternative="two-sided")
    u_new, p_new = mannwhitneyu(guards["debiased_rank"], bigs["debiased_rank"], alternative="two-sided")
    print(f"  Old (consensus) ranks: U = {u_old:.1f}, p = {p_old:.4f}")
    print(f"  New (debiased) ranks: U = {u_new:.1f}, p = {p_new:.4f}")
    print("  Interpretation: Lower p suggests position bias; debiasing typically increases p.")


def fun_awards(stats: pd.DataFrame, debiased: pd.DataFrame):
    section_header("10. FUN AWARDS")
    merged = stats.merge(
        debiased[["player_name", "position", "debiased_rank"]],
        on=["player_name", "position"],
        how="left",
    )

    # MVP: #1 debiased
    mvp = merged.loc[merged["debiased_rank"].idxmin()]
    print(f"\n  MVP: {mvp['player_name']} ({mvp['position']}) — #1 overall (debiased)")

    # DPOY: best composite of def_rating (lower better) + bpg + spg (higher better)
    z_def = (merged["def_rating"].max() - merged["def_rating"]) / (merged["def_rating"].std() or 1)
    z_bpg = (merged["bpg"] - merged["bpg"].mean()) / (merged["bpg"].std() or 1)
    z_spg = (merged["spg"] - merged["spg"].mean()) / (merged["spg"].std() or 1)
    merged["def_composite"] = z_def + z_bpg + z_spg
    dpoy = merged.loc[merged["def_composite"].idxmax()]
    print(f"  Defensive Player of the Year: {dpoy['player_name']} ({dpoy['position']}) — elite DefRtg + BPG + SPG")

    # Most Improved Potential: high VORP, low PPG (efficient hidden gem)
    low_ppg = merged[merged["ppg"] <= merged["ppg"].median()]
    if not low_ppg.empty:
        mip = low_ppg.loc[low_ppg["vorp"].idxmax()]
        print(f"  Most Improved Potential: {mip['player_name']} ({mip['position']}) — VORP {mip['vorp']:.1f}, PPG {mip['ppg']:.1f} (efficient hidden gem)")

    # Most Overrated: high PPG, worst plus_minus
    high_ppg = merged[merged["ppg"] >= merged["ppg"].quantile(0.80)]
    if not high_ppg.empty:
        mo = high_ppg.loc[high_ppg["plus_minus"].idxmin()]
        print(f"  Most Overrated: {mo['player_name']} ({mo['position']}) — PPG {mo['ppg']:.1f}, +/- {mo['plus_minus']:.1f}")

    # Best Teammate: high plus_minus, below-average PPG
    avg_ppg = merged["ppg"].mean()
    below_avg = merged[merged["ppg"] < avg_ppg]
    if not below_avg.empty:
        bt = below_avg.loc[below_avg["plus_minus"].idxmax()]
        print(f"  Best Teammate: {bt['player_name']} ({bt['position']}) — +/- {bt['plus_minus']:.1f}, PPG {bt['ppg']:.1f} (below avg scoring)")

    # Unicorn: most positionally unusual (guard in top rpg/bpg, center in top apg/spg/3P%/FT%, etc.)
    # Simple: max |z-score| for a stat where position typically is weak
    pos_weak = {
        "PG": "rpg", "SG": "rpg", "SF": "apg", "PF": "apg",
        "C": "apg",  # C with high apg is unicorn
    }
    unicorn_score = pd.Series(0.0, index=merged.index)
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        mask = merged["position"] == pos
        weak_stat = pos_weak.get(pos, "vorp")
        grp = merged.loc[mask, weak_stat]
        z = (grp - grp.mean()) / (grp.std() or 1)
        unicorn_score.loc[mask] = z
    merged["unicorn_z"] = unicorn_score
    uni = merged.loc[merged["unicorn_z"].idxmax()]
    print(f"  Unicorn Award: {uni['player_name']} ({uni['position']}) — most positionally unusual stat line (e.g. {pos_weak.get(uni['position'], 'impact')})")


def main():
    stats, rankings, debiased = load_data()

    print("\n" + "#" * 70)
    print("#  NBA PLAYER RANKING ENGINE — FINAL REPORT")
    print("#" * 70)

    league_summary(stats)
    stat_kings(stats)
    outlier_spotlight(stats)
    top25_debiased(debiased)
    all_nba_teams(debiased)
    most_well_rounded(stats)
    position_rankings_summary(debiased)
    method_agreement(rankings, debiased)
    bias_report(debiased)
    fun_awards(stats, debiased)

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
