"""
Exploratory Data Analysis for synthetic NBA player stats.
Loads data from data/nba_synthetic_stats.csv and produces publication-quality plots.
"""

import os
import numpy as np
import pandas as pd

# Configuration
DATA_PATH = "data/nba_synthetic_stats.csv"

# Stat columns for distributions (all numeric except player_name, position)
STAT_COLS = [
    "ppg", "rpg", "apg", "bpg", "spg", "plus_minus",
    "def_rating", "off_rating", "fg_pct", "three_pt_pct", "ft_pct", "vorp",
]


def load_data():
    """Load the synthetic NBA stats dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def print_insights(df):
    """Print key EDA insights: correlations, outliers, and average stats by position."""
    numeric = df[STAT_COLS]
    corr = numeric.corr()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # 1. Most correlated pairs (excluding diagonal and duplicates)
    print("\n1. STRONGEST CORRELATIONS (|r| > 0.5):")
    pairs = []
    for i in range(len(STAT_COLS)):
        for j in range(i + 1, len(STAT_COLS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                pairs.append((STAT_COLS[i], STAT_COLS[j], r))
    pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, r in pairs[:15]:
        print(f"   {a} <-> {b}: r = {r:.3f}")

    # 2. Notable outliers (using IQR method)
    print("\n2. NOTABLE OUTLIERS (IQR method, beyond 1.5*IQR):")
    for col in STAT_COLS:
        q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < low) | (df[col] > high)]
        if len(outliers) > 0:
            for _, row in outliers.iterrows():
                print(f"   {col}: {row['player_name']} ({row['position']}) = {row[col]}")

    # 3. Average stats by position
    print("\n3. AVERAGE STATS BY POSITION:")
    by_pos = df.groupby("position")[STAT_COLS].agg("mean").round(3)
    by_pos = by_pos.reindex(["PG", "SG", "SF", "PF", "C"])
    print(by_pos.to_string())


def main():
    print("Loading data from", DATA_PATH)
    df = load_data()
    print(f"Loaded {len(df)} players, {len(df.columns)} columns.\n")

    print_insights(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
