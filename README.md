# NBA Player Ranking Engine: A Multi-Model Statistical Analysis

A Python project that generates synthetic NBA-style player statistics, ranks players using multiple statistical methods (including position-adjusted and debiased approaches), and produces exploratory analyses, leaderboards, and a comprehensive final report.

## What This Project Does

1. **Generates** a realistic synthetic dataset of 150 NBA players with correlated stats (points, rebounds, assists, blocks, steals, shooting percentages, VORP, plus/minus, etc.), including **injected outliers** (e.g. Point Centers, Rebounding Guards, Shot-blocking Wings, Empty Stats Guy, Quiet MVP) and **lower league-wide scoring** (league avg PPG ~11–13, cap at 33).
2. **Explores** the data with correlation heatmaps, distribution plots, box plots by position, and scatter matrices.
3. **Ranks** every player using five global methods: weighted composite, PCA, K-Means tiers, percentile-based composite, and a Bayesian (Bradley–Terry–inspired) matchup rating; then a **debiased** ranking using **position-relative z-scores** to remove bias toward centers and power forwards.
4. **Visualizes** results with rank-comparison heatmaps, bump charts, radar profiles, tier distribution, and ESPN-style leaderboards.
5. **Stat leaderboards**: Top 10 per stat, “stat kings,” positional surprises, and most well-rounded players (top-10 appearances).
6. **Position rankings**: Per-position leaderboards with position-specific weights, All-NBA First/Second/Third Teams, and global vs position-rank comparison.
7. **Final report**: A single console report summarizing league stats, stat kings, outliers, top 25 debiased, All-NBA teams, method agreement, bias analysis, and “fun awards.”

All outputs are written to `data/` (CSVs) and `plots/` (PNG figures).

---

## Pipeline (main.py)

Run the full pipeline (8 steps):

```bash
pip install -r requirements.txt
python main.py
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `generate_dataset.py` | Synthetic dataset with outliers and lower scoring |
| 2 | `eda.py` | Exploratory data analysis and EDA plots |
| 3 | `ranking_models.py` | Five ranking methods + consensus, PCA/K-Means plots |
| 4 | `visualize_rankings.py` | Ranking dashboard (heatmaps, bump, radar, top 10) |
| 5 | `stat_leaderboards.py` | Per-stat top 10, stat kings, positional surprises, top-10 appearances |
| 6 | `position_rankings.py` | Position-adjusted rankings, All-NBA teams, global vs position |
| 7 | `debiased_rankings.py` | Debiased global ranking (position-relative z-scores), bias plots |
| 8 | `final_report.py` | Comprehensive final console report |

---

## Statistical Methods

| Method | Description |
|--------|-------------|
| **Weighted composite** | Expert weights; stats Z-score normalized (def_rating inverted); rank by weighted sum. |
| **PCA** | Standardize stats, PCA; rank by PC1 (90% variance). |
| **K-Means tiers** | Cluster into 4–5 tiers; rank by tier then distance to best centroid. |
| **Percentile-based** | Percentile (0–100) per stat, same weights; robust to outliers. |
| **Bayesian rating** | Simulated matchups from weighted composite; rating = (wins+1)/(wins+losses+2). |
| **Consensus** | Average of the five ranks above, re-ranked. |
| **Debiased** | Z-scores computed **within position**; same expert weights; removes bias toward bigs. |

---

## Outlier System (generate_dataset.py)

After base generation, 15–20 players are overwritten with **archetype** profiles and memorable names:

- **Point Center** (2–3): C with 7–11 APG, high RPG, lower BPG.
- **Rebounding Guard** (2–3): PG/SG with 8–12 RPG, 5+ APG.
- **Shot-blocking Wing** (2): SF/SG with 2.5–4.0 BPG, 6–9 RPG.
- **Scoring Big with Range** (2–3): C/PF with 38%+ 3P, 20+ PPG.
- **Defensive Guard Menace** (2–3): PG/SG with 2.2–3.0 SPG, elite DefRtg.
- **Empty Stats Guy** (2–3): High PPG, bad +/-, DefRtg, FG%, negative VORP.
- **Quiet MVP** (1–2): Strong +/-, DefRtg, VORP, FG%; not top PPG.

League-wide PPG is reduced (~8–12%) so that only 5–8 players are above 25 PPG and no one above 33.

---

## Position-Adjusted Rankings (position_rankings.py)

- **Weights** differ by position (e.g. PG: high APG/SPG, low RPG/BPG; C: high RPG/BPG, lower APG/3P%).
- Stats are **z-scored within position** so players are compared to peers.
- **All-NBA First/Second/Third Teams**: #1, #2, #3 at each position by position-adjusted rank.
- Compares **global consensus rank** vs **within-position rank** to flag overrated/underrated players.

---

## Debiasing (debiased_rankings.py)

- **Problem:** League-wide z-scores favor centers (high RPG/BPG) and underweight guard strengths (APG, 3P%, SPG).
- **Method:** Z-score each stat **within position**; flip def_rating; apply the same expert weights; rank globally.
- **Output:** `data/debiased_rankings.csv` and plots (rank change distribution, old vs new ranks, position bias comparison).
- **Bias analysis:** Average rank change by position; Mann–Whitney U (guards vs bigs) on old vs new ranks.

---

## Project Structure

```
Mock NBA Ranking Project/
├── main.py                    # Master script: runs all 8 steps
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── generate_dataset.py        # Step 1: synthetic data + outliers
├── eda.py                     # Step 2: EDA and EDA plots
├── ranking_models.py          # Step 3: 5 methods + consensus
├── visualize_rankings.py      # Step 4: ranking dashboard
├── stat_leaderboards.py       # Step 5: stat leaderboards + top-10 appearances
├── position_rankings.py       # Step 6: position rankings + All-NBA teams
├── debiased_rankings.py       # Step 7: debiased ranking + bias plots
├── final_report.py            # Step 8: comprehensive final report
├── data/
│   ├── nba_synthetic_stats.csv    # Raw stats (with archetype)
│   ├── final_rankings.csv         # Global ranks (5 methods + consensus)
│   └── debiased_rankings.csv      # Debiased ranks + full stats
└── plots/
    ├── correlation_heatmap.png
    ├── stat_distributions.png
    ├── position_boxplots.png
    ├── scatter_matrix.png
    ├── top_scorers.png
    ├── pca_scree.png
    ├── pca_biplot.png
    ├── kmeans_elbow.png
    ├── kmeans_clusters.png
    ├── rank_comparison_heatmap.png
    ├── rank_correlation.png
    ├── bump_chart.png
    ├── top5_radar.png
    ├── tier_distribution.png
    ├── top10_summary.png
    ├── stat_leaders_grid.png
    ├── top10_appearances.png
    ├── position_top10_PG.png ... position_top10_C.png
    ├── all_nba_teams.png
    ├── global_vs_position_rank.png
    ├── rank_change_distribution.png
    ├── old_vs_new_ranks.png
    └── position_bias_comparison.png
```

---

## Example Output

- **Top 25 (debiased)** and **All-NBA Teams** are in the final report and in `plots/top10_summary.png` and `plots/all_nba_teams.png`.
- **Final report** (`python final_report.py`) prints: league summary, stat kings, outlier spotlight, top 25 debiased, All-NBA teams, most well-rounded, position top 5, method agreement, bias report, and fun awards (MVP, DPOY, Most Improved Potential, Most Overrated, Best Teammate, Unicorn).

---

## Limitations and Future Work

- **Synthetic data only:** Use real data (e.g. Basketball-Reference, NBA API) for real-world analysis.
- **Advanced metrics:** Add PER, win shares, BPM, usage rate.
- **Time series:** Rolling averages, rank trends over seasons.
- **Debiasing:** Try other position-relative schemes or regression-based adjustments.

---

## License

Portfolio/educational project. No affiliation with the NBA or any official data source.
