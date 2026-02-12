"""
Visual dashboard for NBA player rankings.
Loads final_rankings.csv and nba_synthetic_stats.csv.
Note: All plot generation has been moved to create_plots.py
"""

import os
import numpy as np
import pandas as pd

# Paths
RANKINGS_PATH = "data/final_rankings.csv"
DEBIASED_PATH = "data/debiased_rankings.csv"
STATS_PATH = "data/nba_synthetic_stats.csv"


def load_data():
    """Load rankings, debiased rankings, and stats."""
    rankings = pd.read_csv(RANKINGS_PATH)
    debiased = pd.read_csv(DEBIASED_PATH)
    stats = pd.read_csv(STATS_PATH)
    merged = rankings.merge(stats, on=["player_name", "position"], how="left")
    return rankings, debiased, stats, merged


def main():
    print("Loading data...")
    rankings, debiased, stats, merged = load_data()
    print(f"Loaded {len(rankings)} rankings, {len(debiased)} debiased rankings, {len(stats)} stats.")
    print("\nNote: All plot generation has been moved to create_plots.py")


if __name__ == "__main__":
    main()
