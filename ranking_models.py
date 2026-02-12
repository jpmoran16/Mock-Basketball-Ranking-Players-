"""
Rank NBA players using five statistical methods.
Loads data from data/nba_synthetic_stats.csv and produces final_rankings with consensus.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuration
DATA_PATH = "data/nba_synthetic_stats.csv"
RNG = np.random.default_rng(42)

STAT_COLS = [
    "ppg", "rpg", "apg", "bpg", "spg", "plus_minus",
    "def_rating", "off_rating", "fg_pct", "three_pt_pct", "ft_pct", "vorp",
]

# Expert weights (must sum to 1.0)
EXPERT_WEIGHTS = {
    "ppg": 0.21,
    "rpg": 0.06,
    "apg": 0.13,
    "bpg": 0.05,
    "spg": 0.05,
    "plus_minus": 0.10,
    "def_rating": 0.07,
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
    print("ranking_models.py: Global weights updated")
    print("  Top 3 changes: vorp 0.11->0.16 (+0.05), apg 0.10->0.13 (+0.03), ppg 0.20->0.21 (+0.01)")


def load_and_prepare():
    """Load dataset and return (df, X_inverted) where def_rating is flipped for 'higher is better'."""
    df = pd.read_csv(DATA_PATH)
    X = df[STAT_COLS].copy()
    # Lower def_rating is better → multiply by -1 so higher is better
    X["def_rating"] = -X["def_rating"]
    return df, X


def method1_weighted_composite(df, X):
    """Method 1: Z-score normalize, then weighted sum. Rank by composite (higher = better)."""
    # Z-score normalize
    mean = X.mean()
    std = X.std()
    # Avoid division by zero
    std = std.replace(0, 1)
    Z = (X - mean) / std
    # Weighted sum: use order of STAT_COLS and EXPERT_WEIGHTS
    w = np.array([EXPERT_WEIGHTS[c] for c in STAT_COLS])
    composite = (Z * w).sum(axis=1).values
    # Rank: higher composite = better = rank 1
    rank = pd.Series(composite).rank(ascending=False, method="min").astype(int).values
    return rank, composite


def method3_kmeans(df, X_scaled, optimal_k, tier_names):
    """K-Means with optimal k; label tiers by centroid quality; rank by tier then distance to superstar centroid."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    # Centroid "quality": use sum of coordinates (scaled space) or first PC of centroids
    # Simpler: mean of centroid along each feature (all higher = better after our scaling)
    centroid_quality = centroids.mean(axis=1)
    # Map cluster index to tier: highest quality = Superstar (0), then All-Star (1), etc.
    tier_order = np.argsort(-centroid_quality)  # cluster index for Superstar, All-Star, ...
    cluster_to_tier_rank = {tier_order[i]: i for i in range(optimal_k)}
    superstar_centroid = centroids[tier_order[0]]
    # Per-player: tier rank (0=Superstar, 1=All-Star, ...) and distance to superstar centroid
    tier_rank = np.array([cluster_to_tier_rank[l] for l in labels])
    dist_to_superstar = np.linalg.norm(X_scaled - superstar_centroid, axis=1)
    # Global rank: sort by (tier_rank, dist_to_superstar)
    # Lower tier_rank first; within same tier, lower distance first
    rank = pd.Series(zip(tier_rank, dist_to_superstar)).rank(method="dense", ascending=True).astype(int).values
    # Actually rank: we want (tier_rank asc, dist asc). Lexicographic sort.
    sort_keys = list(zip(tier_rank, dist_to_superstar))
    order = np.lexsort((dist_to_superstar, tier_rank))
    rank = np.empty(len(df), dtype=int)
    rank[order] = np.arange(1, len(df) + 1)
    # Tier labels for plot
    tier_label = np.array([tier_names[cluster_to_tier_rank[l]] for l in labels])
    return rank, labels, tier_label, kmeans


def method4_percentile(df, X):
    """Method 4: Percentile rank (0-100) per stat, then same expert weights. Higher percentile = better."""
    # Percentile of each value within the column (ascending: higher value = higher percentile)
    P = X.apply(lambda col: col.rank(pct=True).values * 100, axis=0)
    w = np.array([EXPERT_WEIGHTS[c] for c in STAT_COLS])
    composite = (P * w).sum(axis=1).values
    rank = pd.Series(composite).rank(ascending=False, method="min").astype(int).values
    return rank


def method5_bayesian(df, composite_scores, k=1.5, n_matchups=1000):
    """Method 5: Simulate n_matchups per player; P(A beats B) = 1/(1+exp(-k*(sA-sB))). Rating = (w+1)/(w+l+2)."""
    n = len(df)
    wins = np.zeros(n)
    losses = np.zeros(n)
    for i in range(n):
        opponents = RNG.integers(0, n, size=n_matchups)
        for j in opponents:
            if j == i:
                continue
            diff = composite_scores[i] - composite_scores[j]
            p_win = 1.0 / (1.0 + np.exp(-k * diff))
            if RNG.random() < p_win:
                wins[i] += 1
            else:
                losses[i] += 1
    # Bayesian rating: (wins+1)/(wins+losses+2)
    rating = (wins + 1) / (wins + losses + 2)
    rank = pd.Series(rating).rank(ascending=False, method="min").astype(int).values
    return rank


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading data...")
    df, X = load_and_prepare()
    n = len(df)

    # ----- Method 1: Weighted Composite -----
    rank1, composite = method1_weighted_composite(df, X)
    print("Method 1 (Weighted Composite): done.")

    # ----- Method 2: PCA -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Full PCA for scree plot (explained variance)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumvar, 0.9)) + 1
    n_components_90 = min(max(1, n_components_90), len(STAT_COLS))
    # Fit PCA with n_components for 90% variance; use PC1 for ranking
    pca = PCA(n_components=n_components_90)
    X_pca_full = pca.fit_transform(X_scaled)
    pc1_score = X_pca_full[:, 0]
    rank2 = pd.Series(pc1_score).rank(ascending=False, method="min").astype(int).values

    # 2D PCA for later visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    print("Method 2 (PCA): done.")

    # ----- Method 3: K-Means -----
    optimal_k = 5  # Fixed k=5 tiers
    tier_names = ["Superstar", "All-Star", "Starter", "Rotation", "Bench"][:optimal_k]
    rank3, kmeans_labels, tier_label, kmeans = method3_kmeans(df, X_scaled, optimal_k, tier_names)
    print("Method 3 (K-Means): done.")

    # ----- Method 4: Percentile -----
    rank4 = method4_percentile(df, X)
    print("Method 4 (Percentile): done.")

    # ----- Method 5: Bayesian -----
    rank5 = method5_bayesian(df, composite, k=1.5, n_matchups=1000)
    print("Method 5 (Bayesian): done.")

    # ----- Final rankings DataFrame -----
    final_rankings = pd.DataFrame({
        "player_name": df["player_name"],
        "position": df["position"],
        "rank_weighted": rank1,
        "rank_pca": rank2,
        "rank_kmeans": rank3,
        "rank_percentile": rank4,
        "rank_bayesian": rank5,
    })
    avg_rank = final_rankings[
        ["rank_weighted", "rank_pca", "rank_kmeans", "rank_percentile", "rank_bayesian"]
    ].mean(axis=1)
    final_rankings["consensus_rank"] = avg_rank.rank(method="min").astype(int)
    final_rankings = final_rankings.sort_values("consensus_rank").reset_index(drop=True)

    # Top 25 by consensus
    print("\n" + "=" * 80)
    print("TOP 25 PLAYERS BY CONSENSUS RANKING")
    print("=" * 80)
    top25 = final_rankings.head(25)[
        ["consensus_rank", "player_name", "position", "rank_weighted", "rank_pca", "rank_kmeans", "rank_percentile", "rank_bayesian"]
    ]
    print(top25.to_string(index=False))

    # Spearman correlation between methods
    rank_cols = ["rank_weighted", "rank_pca", "rank_kmeans", "rank_percentile", "rank_bayesian"]
    spearman = final_rankings[rank_cols].corr(method="spearman")
    print("\n" + "=" * 80)
    print("SPEARMAN RANK CORRELATION BETWEEN RANKING METHODS")
    print("=" * 80)
    print(spearman.round(3).to_string())

    # Save
    out_path = "data/final_rankings.csv"
    final_rankings.to_csv(out_path, index=False)
    print(f"\nFull rankings saved to {out_path}.")


if __name__ == "__main__":
    main()
