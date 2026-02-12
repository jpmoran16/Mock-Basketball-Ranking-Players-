"""
Unified plot generation for NBA Player Ranking Project.
All visualizations use a consistent dark theme with position-based color palette.
Generates 23 total plots: 11 main plots + 12 individual stat top-10 charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import position_rankings as pr

# ============================================================================
# CONFIGURATION & STYLE GUIDE
# ============================================================================

# Color Palette
BG_DARK = "#1a1a2e"  # Dark charcoal background
BG_PLOT = "#16213e"  # Slightly lighter plot area
TEXT_WHITE = "#ffffff"
TEXT_GRAY = "#b0b0b0"
# Position colors (single source of truth)
POSITION_COLORS = {
    "PG": "#d21404",   # Candy red
    "SG": "#cc5801",   # Yam orange
    "SF": "#ffe135",   # Banana yellow
    "PF": "#00a86b",   # Jade green
    "C": "#0e4c92",    # Yale blue
}
GOLD = "#ffd700"
SILVER = "#c0c0c0"
BRONZE = "#cd7f32"

# Font settings
FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]
TITLE_SIZE = 20
SUBTITLE_SIZE = 13
LABEL_SIZE = 11

# Paths
STATS_PATH = "data/nba_synthetic_stats.csv"
RANKINGS_PATH = "data/final_rankings.csv"
DEBIASED_PATH = "data/debiased_rankings.csv"
PLOTS_DIR = "plots"
DPI = 200

# Stat columns
STAT_COLS = [
    "ppg", "rpg", "apg", "bpg", "spg", "plus_minus",
    "def_rating", "off_rating", "fg_pct", "three_pt_pct", "ft_pct", "vorp",
]

STAT_LABELS = {
    "ppg": "Points",
    "rpg": "Rebounds",
    "apg": "Assists",
    "bpg": "Blocks",
    "spg": "Steals",
    "plus_minus": "+/-",
    "def_rating": "Def Rtg",
    "off_rating": "Off Rtg",
    "fg_pct": "FG%",
    "three_pt_pct": "3P%",
    "ft_pct": "FT%",
    "vorp": "VORP",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_dark_style(fig, ax):
    """Apply dark theme styling to figure and axes."""
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PLOT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_GRAY)
    ax.spines['left'].set_color(TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.xaxis.label.set_color(TEXT_GRAY)
    ax.yaxis.label.set_color(TEXT_GRAY)
    for label in ax.get_xticklabels():
        label.set_color(TEXT_GRAY)
    for label in ax.get_yticklabels():
        label.set_color(TEXT_GRAY)


def get_position_color(pos):
    """Get color for a position."""
    return POSITION_COLORS.get(pos, TEXT_GRAY)


def add_position_legend(ax, loc="upper right"):
    """Add position color legend with small circles and semi-transparent background."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POSITION_COLORS[p],
               markersize=8, markeredgecolor=TEXT_GRAY, markeredgewidth=0.8, label=p)
        for p in ["PG", "SG", "SF", "PF", "C"]
    ]
    leg = ax.legend(handles=handles, title="Position", loc=loc, framealpha=0.85,
                    facecolor=BG_PLOT, edgecolor=TEXT_GRAY, title_fontsize=10, fontsize=9)
    leg.get_frame().set_boxstyle("round,pad=0.3")


# ============================================================================
# PLOT 1: Top 10 Overall Players
# ============================================================================

def plot_top10_overall():
    """Top 10 players by debiased rank - horizontal bar chart."""
    debiased = pd.read_csv(DEBIASED_PATH)
    top10 = debiased.nsmallest(10, "debiased_rank").sort_values("debiased_rank")
    scores = top10["debiased_score"].values
    x_max = scores.max()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplots_adjust(right=0.88)
    setup_dark_style(fig, ax)
    
    n = len(top10)
    y_pos = np.arange(n)
    # Bar heights: #1 slightly taller (0.65), rest 0.6
    heights = np.full(n, 0.6)
    heights[0] = 0.65
    colors = [get_position_color(pos) for pos in top10["position"]]
    bars = ax.barh(y_pos, scores, height=heights, color=colors, alpha=0.8, edgecolor=TEXT_GRAY, linewidth=1)
    
    # Rank numbers: fixed position to the left of bars (data coords)
    x_min = -x_max * 0.15
    ax.set_xlim(x_min, x_max * 1.35)
    for i in range(n):
        ax.text(x_min + 0.02 * x_max, i, f"#{i+1}", fontsize=22, fontweight="bold",
                color=GOLD, va="center", ha="left")
    
    # Player name and position inside bar; stat line outside
    bar_start_pct = 0.02
    name_start = bar_start_pct * x_max
    for i, (idx, row) in enumerate(top10.iterrows()):
        ax.text(name_start, i, row["player_name"], fontsize=14, fontweight="bold",
                color=TEXT_WHITE, va="center", ha="left")
        pos_color = get_position_color(row["position"])
        badge_x = name_start + 0.06 * x_max
        circle = Circle((badge_x, i), 0.18, color=pos_color, zorder=10)
        ax.add_patch(circle)
        ax.text(badge_x, i, row["position"], fontsize=9, fontweight="bold", color=BG_DARK,
                va="center", ha="center", zorder=11)
        # Stat line outside bar (right side) - use data coords with margin
        stat_line = f"{row['ppg']:.1f} PPG | {row['rpg']:.1f} RPG | {row['apg']:.1f} APG | {row['vorp']:.1f} VORP"
        ax.text(x_max * 1.02, i, stat_line, fontsize=10, color=TEXT_GRAY, va="center", ha="left")
    
    ax.text(x_max * 1.02, 0, "★", fontsize=28, color=GOLD, va="center", ha="left", fontweight="bold")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Debiased Score", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_title("TOP 10 PLAYERS", fontsize=TITLE_SIZE, fontweight="bold", color=TEXT_WHITE, pad=20)
    ax.axhline(y=-0.5, xmin=0.1, xmax=0.9, color=GOLD, linewidth=3, clip_on=False)
    add_position_legend(ax, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "top10_overall.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/top10_overall.png")


# ============================================================================
# PLOT 2: Top 5 MVP Candidates
# ============================================================================

def plot_top5_mvp():
    """Card-style layout for top 5 MVP candidates."""
    debiased = pd.read_csv(DEBIASED_PATH)
    # debiased_rankings.csv already contains all stat columns
    
    # MVP composite: ppg*0.30 + apg*0.15 + vorp*0.25 + plus_minus*0.15 + off_rating*0.10 + fg_pct*0.05
    debiased["mvp_score"] = (
        debiased["ppg"] * 0.30 +
        debiased["apg"] * 0.15 +
        debiased["vorp"] * 0.25 +
        debiased["plus_minus"] * 0.15 +
        debiased["off_rating"] * 0.10 +
        debiased["fg_pct"] * 0.05
    )
    top5 = debiased.nlargest(5, "mvp_score")
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)
    
    ax.set_title("MVP RACE", fontsize=28, fontweight="bold", color=GOLD, pad=20)
    
    card_width = 0.18
    card_height = 0.7
    x_start = 0.05
    y_center = 0.5
    
    for i, (idx, row) in enumerate(top5.iterrows()):
        x = x_start + i * 0.19
        is_mvp = (i == 0)
        
        # Card background
        card = FancyBboxPatch(
            (x, y_center - card_height/2), card_width, card_height,
            boxstyle="round,pad=0.02", 
            facecolor=BG_PLOT,
            edgecolor=GOLD if is_mvp else get_position_color(row["position"]),
            linewidth=4 if is_mvp else 2,
            transform=ax.transAxes
        )
        ax.add_patch(card)
        
        # Rank
        rank_text = f"#{i+1}" if i > 0 else "#1 MVP"
        ax.text(x + card_width/2, y_center + 0.25, rank_text,
                fontsize=16 if is_mvp else 14, fontweight="bold", color=GOLD if is_mvp else TEXT_WHITE,
                ha="center", va="center", transform=ax.transAxes)
        
        # Player name
        ax.text(x + card_width/2, y_center + 0.1, row["player_name"],
                fontsize=14 if is_mvp else 12, fontweight="bold", color=TEXT_WHITE,
                ha="center", va="center", transform=ax.transAxes)
        
        # Position badge
        pos_color = get_position_color(row["position"])
        ax.text(x + card_width/2, y_center - 0.05, row["position"],
                fontsize=11, fontweight="bold", color=pos_color,
                ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK, edgecolor=pos_color, linewidth=1.5))
        
        # Stats
        stats_text = (
            f"PPG: {row['ppg']:.1f}\n"
            f"APG: {row['apg']:.1f}\n"
            f"VORP: {row['vorp']:.1f}\n"
            f"+/-: {row['plus_minus']:+.1f}\n"
            f"Off Rtg: {row['off_rating']:.1f}\n"
            f"FG%: {row['fg_pct']:.3f}"
        )
        ax.text(x + card_width/2, y_center - 0.35, stats_text,
                fontsize=9, color=TEXT_GRAY, ha="center", va="top",
                transform=ax.transAxes, linespacing=1.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "top5_mvp.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/top5_mvp.png")


# ============================================================================
# PLOT 3: Top 5 DPOY Candidates
# ============================================================================

def plot_top5_dpoy():
    """Card-style layout for top 5 DPOY candidates."""
    stats = pd.read_csv(STATS_PATH)
    
    # Z-score stats for DPOY composite
    def_rating_inv = -stats["def_rating"]  # Lower is better, so invert
    def_rating_z = (def_rating_inv - def_rating_inv.mean()) / def_rating_inv.std()
    bpg_z = (stats["bpg"] - stats["bpg"].mean()) / stats["bpg"].std()
    spg_z = (stats["spg"] - stats["spg"].mean()) / stats["spg"].std()
    pm_z = (stats["plus_minus"] - stats["plus_minus"].mean()) / stats["plus_minus"].std()
    
    stats["dpoy_score"] = (
        def_rating_z * 0.35 +
        bpg_z * 0.25 +
        spg_z * 0.20 +
        pm_z * 0.20
    )
    top5 = stats.nlargest(5, "dpoy_score")
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)
    
    ax.set_title("DEFENSIVE PLAYER OF THE YEAR", fontsize=28, fontweight="bold", color=SILVER, pad=20)
    
    card_width = 0.18
    card_height = 0.7
    x_start = 0.05
    y_center = 0.5
    
    for i, (idx, row) in enumerate(top5.iterrows()):
        x = x_start + i * 0.19
        is_dpoy = (i == 0)
        
        # Card with silver/steel border for DPOY
        card = FancyBboxPatch(
            (x, y_center - card_height/2), card_width, card_height,
            boxstyle="round,pad=0.02",
            facecolor=BG_PLOT,
            edgecolor=SILVER if is_dpoy else "#708090",  # Steel gray for others
            linewidth=4 if is_dpoy else 2,
            transform=ax.transAxes
        )
        ax.add_patch(card)
        
        rank_text = f"#{i+1}" if i > 0 else "#1 DPOY"
        ax.text(x + card_width/2, y_center + 0.25, rank_text,
                fontsize=16 if is_dpoy else 14, fontweight="bold", color=SILVER,
                ha="center", va="center", transform=ax.transAxes)
        
        ax.text(x + card_width/2, y_center + 0.1, row["player_name"],
                fontsize=14 if is_dpoy else 12, fontweight="bold", color=TEXT_WHITE,
                ha="center", va="center", transform=ax.transAxes)
        
        pos_color = get_position_color(row["position"])
        ax.text(x + card_width/2, y_center - 0.05, row["position"],
                fontsize=11, fontweight="bold", color=pos_color,
                ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK, edgecolor=pos_color, linewidth=1.5))
        
        stats_text = (
            f"Def Rtg: {row['def_rating']:.1f}\n"
            f"BPG: {row['bpg']:.1f}\n"
            f"SPG: {row['spg']:.1f}\n"
            f"+/-: {row['plus_minus']:+.1f}\n"
            f"Score: {row['dpoy_score']:.2f}"
        )
        ax.text(x + card_width/2, y_center - 0.35, stats_text,
                fontsize=9, color=TEXT_GRAY, ha="center", va="top",
                transform=ax.transAxes, linespacing=1.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "top5_dpoy.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/top5_dpoy.png")


# ============================================================================
# PLOT 4: All-NBA Teams
# ============================================================================

def plot_all_nba_teams():
    """All-NBA First, Second, Third Team lineup graphic."""
    stats = pd.read_csv(STATS_PATH)
    rankings = pd.read_csv(RANKINGS_PATH)
    # Merge rankings with stats
    merged = rankings.merge(stats, on=["player_name", "position"], how="left")
    merged = pr.compute_position_scores(merged)
    teams = pr.build_all_nba_teams(merged)
    
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)
    
    ax.set_title("ALL-NBA TEAMS", fontsize=32, fontweight="bold", color=TEXT_WHITE, pad=30)
    
    team_configs = [
        ("First Team", GOLD, 0.85),
        ("Second Team", SILVER, 0.50),
        ("Third Team", BRONZE, 0.15),
    ]
    
    pos_order = ["PG", "SG", "SF", "PF", "C"]
    slot_width = 0.18
    card_height = 0.16
    x_start = 0.05
    STAT_BRIGHT = "#d0d0d0"
    
    for team_name, accent_color, y_base in team_configs:
        players = teams[team_name]
        ax.text(0.5, y_base + 0.08, f"ALL-NBA {team_name.upper()}",
                fontsize=20, fontweight="bold", color=accent_color,
                ha="center", va="center", transform=ax.transAxes)
        ax.plot([0.05, 0.95], [y_base - 0.02, y_base - 0.02],
                color=accent_color, linewidth=2, alpha=0.5, transform=ax.transAxes)
        
        for j, pos in enumerate(pos_order):
            player = next((p for p in players if p["position"] == pos), None)
            if player is None:
                continue
            x = x_start + j * slot_width
            card = FancyBboxPatch(
                (x, y_base - card_height), slot_width - 0.01, card_height,
                boxstyle="round,pad=0.005",
                facecolor=BG_PLOT,
                edgecolor=get_position_color(pos),
                linewidth=2,
                transform=ax.transAxes
            )
            ax.add_patch(card)
            ax.text(x + slot_width/2 - 0.01, y_base - 0.025, pos,
                    fontsize=11, fontweight="bold", color=get_position_color(pos),
                    ha="center", va="bottom", transform=ax.transAxes)
            ax.text(x + slot_width/2 - 0.01, y_base - 0.07, player["player_name"],
                    fontsize=14, fontweight="bold", color=TEXT_WHITE,
                    ha="center", va="center", transform=ax.transAxes)
            stats_line = f"PPG: {player['ppg']:.1f} | RPG: {player['rpg']:.1f} | APG: {player['apg']:.1f} | VORP: {player['vorp']:.1f}"
            ax.text(x + slot_width/2 - 0.01, y_base - 0.115, stats_line,
                    fontsize=11, color=STAT_BRIGHT, ha="center", va="top", transform=ax.transAxes)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "all_nba_teams.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/all_nba_teams.png")


# ============================================================================
# PLOT 5: All-Defensive Teams
# ============================================================================

def plot_all_defensive_teams():
    """All-Defensive First and Second Team."""
    stats = pd.read_csv(STATS_PATH)
    
    # Compute defensive score per position (z-scored within position)
    def compute_defensive_score_per_position():
        result = []
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            df_pos = stats[stats["position"] == pos].copy()
            if df_pos.empty:
                continue
            
            def_rating_inv = -df_pos["def_rating"]
            def_rating_z = (def_rating_inv - def_rating_inv.mean()) / def_rating_inv.std()
            bpg_z = (df_pos["bpg"] - df_pos["bpg"].mean()) / df_pos["bpg"].std()
            spg_z = (df_pos["spg"] - df_pos["spg"].mean()) / df_pos["spg"].std()
            pm_z = (df_pos["plus_minus"] - df_pos["plus_minus"].mean()) / df_pos["plus_minus"].std()
            
            df_pos["def_score"] = def_rating_z * 0.35 + bpg_z * 0.25 + spg_z * 0.20 + pm_z * 0.20
            result.append(df_pos)
        
        return pd.concat(result, ignore_index=True)
    
    df_with_scores = compute_defensive_score_per_position()
    
    # Build teams
    teams = {"First Team": [], "Second Team": []}
    pos_order = ["PG", "SG", "SF", "PF", "C"]
    for team_idx, team_name in enumerate(teams.keys(), start=1):
        for pos in pos_order:
            df_pos = df_with_scores[df_with_scores["position"] == pos].sort_values("def_score", ascending=False)
            if len(df_pos) >= team_idx:
                teams[team_name].append(df_pos.iloc[team_idx - 1])
    
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)
    ax.set_title("ALL-DEFENSIVE TEAMS", fontsize=32, fontweight="bold", color=SILVER, pad=30)
    
    team_configs = [
        ("First Team", SILVER, 0.72),
        ("Second Team", "#708090", 0.32),
    ]
    slot_width = 0.18
    card_height = 0.18
    x_start = 0.05
    STAT_BRIGHT = "#d0d0d0"
    
    for team_name, accent_color, y_base in team_configs:
        players = teams[team_name]
        ax.text(0.5, y_base + 0.08, f"ALL-DEFENSIVE {team_name.upper()}",
                fontsize=20, fontweight="bold", color=accent_color,
                ha="center", va="center", transform=ax.transAxes)
        ax.plot([0.05, 0.95], [y_base - 0.02, y_base - 0.02],
                color=accent_color, linewidth=2, alpha=0.5, transform=ax.transAxes)
        
        for j, pos in enumerate(pos_order):
            player = next((p for p in players if p["position"] == pos), None)
            if player is None:
                continue
            x = x_start + j * slot_width
            card = FancyBboxPatch(
                (x, y_base - card_height), slot_width - 0.01, card_height,
                boxstyle="round,pad=0.005",
                facecolor=BG_PLOT,
                edgecolor=get_position_color(pos),
                linewidth=2,
                transform=ax.transAxes
            )
            ax.add_patch(card)
            ax.text(x + slot_width/2 - 0.01, y_base - 0.03, pos,
                    fontsize=11, fontweight="bold", color=get_position_color(pos),
                    ha="center", va="bottom", transform=ax.transAxes)
            ax.text(x + slot_width/2 - 0.01, y_base - 0.08, player["player_name"],
                    fontsize=14, fontweight="bold", color=TEXT_WHITE,
                    ha="center", va="center", transform=ax.transAxes)
            stats_line = f"DEF: {player['def_rating']:.1f} | BPG: {player['bpg']:.1f} | SPG: {player['spg']:.1f}"
            ax.text(x + slot_width/2 - 0.01, y_base - 0.14, stats_line,
                    fontsize=11, color=STAT_BRIGHT, ha="center", va="top", transform=ax.transAxes)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "all_defensive_teams.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/all_defensive_teams.png")


# ============================================================================
# PLOT 6: Correlation Heatmap
# ============================================================================

def plot_correlation_heatmap():
    """Full correlation matrix with custom diverging colormap."""
    stats = pd.read_csv(STATS_PATH)
    numeric = stats[STAT_COLS]
    corr = numeric.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    setup_dark_style(fig, ax)
    
    # Custom diverging colormap: dark blue -> dark charcoal -> bright red
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#000080", BG_DARK, "#ff0000"]  # Dark blue, charcoal, bright red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom_div", colors, N=n_bins)
    
    # Labels
    labels = [STAT_LABELS.get(col, col.upper()) for col in STAT_COLS]
    
    im = ax.imshow(corr.values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(STAT_COLS)))
    ax.set_yticks(np.arange(len(STAT_COLS)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT_GRAY, fontsize=10)
    ax.set_yticklabels(labels, color=TEXT_GRAY, fontsize=10)
    
    # Annotate with correlation values
    for i in range(len(STAT_COLS)):
        for j in range(len(STAT_COLS)):
            val = corr.iloc[i, j]
            text_color = TEXT_WHITE if abs(val) > 0.5 else TEXT_GRAY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color=text_color, fontsize=9, fontweight="bold" if abs(val) > 0.7 else "normal")
    
    ax.set_title("Stat Correlation Matrix", fontsize=TITLE_SIZE, fontweight="bold",
                 color=TEXT_WHITE, pad=20)
    
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/correlation_heatmap.png")


# ============================================================================
# PLOT 7: Global vs Position Rank Scatter
# ============================================================================

def plot_global_vs_position_rank():
    """Scatter plot comparing global debiased rank vs within-position rank."""
    stats = pd.read_csv(STATS_PATH)
    rankings = pd.read_csv(RANKINGS_PATH)
    debiased = pd.read_csv(DEBIASED_PATH)
    
    # Merge rankings with stats, then add debiased rank
    merged = rankings.merge(stats, on=["player_name", "position"], how="left")
    merged = merged.merge(debiased[["player_name", "debiased_rank"]], on="player_name", how="left")
    merged = pr.compute_position_scores(merged)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    setup_dark_style(fig, ax)
    
    x = merged["debiased_rank"]
    y = merged["position_rank"]
    colors = [get_position_color(pos) for pos in merged["position"]]
    
    ax.scatter(x, y, c=colors, alpha=0.6, s=50, edgecolors=TEXT_GRAY, linewidth=0.5)
    
    # Reference line (scaled)
    max_x = x.max()
    max_y = y.max()
    scale = max_y / max_x
    xs_ref = np.linspace(1, max_x, 100)
    ys_ref = xs_ref * scale
    ax.plot(xs_ref, ys_ref, "--", color=TEXT_GRAY, linewidth=2, alpha=0.5, label="Equal Rank Line")
    
    # Label biggest outliers
    merged["diff"] = (merged["debiased_rank"] - merged["position_rank"]).abs()
    outliers = merged.nlargest(10, "diff")
    for _, row in outliers.iterrows():
        ax.annotate(row["player_name"], (row["debiased_rank"], row["position_rank"]),
                   fontsize=8, color=TEXT_WHITE, alpha=0.9,
                   xytext=(5, 5), textcoords="offset points",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK, edgecolor=get_position_color(row["position"]), alpha=0.8))
    
    # Quadrant labels
    ax.text(0.05, 0.95, "Undervalued by Global", transform=ax.transAxes,
           fontsize=12, color=TEXT_GRAY, ha="left", va="top",
           bbox=dict(boxstyle="round", facecolor=BG_PLOT, alpha=0.7))
    ax.text(0.95, 0.05, "Overvalued by Global", transform=ax.transAxes,
           fontsize=12, color=TEXT_GRAY, ha="right", va="bottom",
           bbox=dict(boxstyle="round", facecolor=BG_PLOT, alpha=0.7))
    
    ax.set_xlabel("Global Debiased Rank (1 = best)", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_ylabel("Within-Position Rank (1 = best)", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_title("Global vs Position Rank Comparison", fontsize=TITLE_SIZE, fontweight="bold",
                 color=TEXT_WHITE, pad=20)
    add_position_legend(ax, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "global_vs_position_rank.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/global_vs_position_rank.png")


# ============================================================================
# PLOT 8: K-Means Clusters
# ============================================================================

def plot_kmeans_clusters():
    """K-Means clusters visualized in PCA space."""
    stats = pd.read_csv(STATS_PATH)
    
    # Replicate PCA and K-Means from ranking_models
    X = stats[STAT_COLS].copy()
    X["def_rating"] = -X["def_rating"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    centroid_quality = centroids.mean(axis=1)
    tier_order = np.argsort(-centroid_quality)
    tier_names = ["Superstar", "All-Star", "Starter", "Rotation", "Bench"]
    cluster_to_tier = {tier_order[i]: tier_names[i] for i in range(5)}
    tier_label = np.array([cluster_to_tier[l] for l in labels])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    setup_dark_style(fig, ax)
    
    tier_colors = {
        "Superstar": GOLD,
        "All-Star": "#ff4444",  # Crimson
        "Starter": "#1e90ff",   # Dodger blue
        "Rotation": "#3cb371",  # Medium sea green
        "Bench": TEXT_GRAY,
    }
    
    tier_sizes = {
        "Superstar": 100,
        "All-Star": 80,
        "Starter": 60,
        "Rotation": 40,
        "Bench": 20,
    }
    
    for tier in tier_names:
        mask = tier_label == tier
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=tier_colors[tier],
                  label=tier, alpha=0.7, s=tier_sizes[tier], edgecolors=TEXT_GRAY, linewidth=0.5)
    
    # Label all Superstars
    superstar_mask = tier_label == "Superstar"
    for idx in np.where(superstar_mask)[0]:
        ax.annotate(stats.iloc[idx]["player_name"], (X_pca[idx, 0], X_pca[idx, 1]),
                   fontsize=8, color=TEXT_WHITE, alpha=0.9,
                   xytext=(5, 5), textcoords="offset points",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_DARK, edgecolor=GOLD, alpha=0.8))
    
    ax.set_xlabel("PC1", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_ylabel("PC2", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_title("K-Means Clusters in PCA Space", fontsize=TITLE_SIZE, fontweight="bold",
                 color=TEXT_WHITE, pad=20)
    ax.legend(title="Tier", facecolor=BG_PLOT, edgecolor=TEXT_GRAY, title_fontsize=12)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "kmeans_clusters.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/kmeans_clusters.png")


# ============================================================================
# PLOT 9: Position Box Plots
# ============================================================================

def plot_position_boxplots():
    """2x2 grid of boxplots for PPG, RPG, APG, BPG by position."""
    stats = pd.read_csv(STATS_PATH)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG_DARK)
    
    box_stats = ["ppg", "rpg", "apg", "bpg"]
    pos_order = ["PG", "SG", "SF", "PF", "C"]
    
    for ax, stat in zip(axes.flat, box_stats):
        setup_dark_style(fig, ax)
        
        data_list = []
        positions_list = []
        for pos in pos_order:
            pos_data = stats[stats["position"] == pos][stat].values
            data_list.append(pos_data)
            positions_list.append(pos)
        
        bp = ax.boxplot(data_list, tick_labels=pos_order, patch_artist=True,
                       showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.5, color=TEXT_GRAY))
        
        for patch, pos in zip(bp['boxes'], pos_order):
            patch.set_facecolor(get_position_color(pos))
            patch.set_alpha(0.7)
            patch.set_edgecolor(TEXT_GRAY)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=TEXT_GRAY)
        
        ax.set_title(f"{STAT_LABELS[stat]} by Position", fontsize=SUBTITLE_SIZE, fontweight="bold",
                    color=TEXT_WHITE, pad=10)
        ax.set_ylabel(STAT_LABELS[stat], color=TEXT_GRAY, fontsize=LABEL_SIZE)
        ax.set_xlabel("Position", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    
    add_position_legend(axes[1, 1], loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "position_boxplots.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/position_boxplots.png")


# ============================================================================
# PLOT 10: PCA Biplot
# ============================================================================

def plot_pca_biplot():
    """PCA biplot with loading vectors and top 15 players labeled."""
    stats = pd.read_csv(STATS_PATH)
    debiased = pd.read_csv(DEBIASED_PATH)
    
    X = stats[STAT_COLS].copy()
    X["def_rating"] = -X["def_rating"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    setup_dark_style(fig, ax)
    
    # Plot players colored by position
    positions = stats["position"]
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        mask = positions == pos
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=get_position_color(pos),
                  label=pos, alpha=0.5, s=30, edgecolors=TEXT_GRAY, linewidth=0.3)
    
    # Loading vectors (arrows)
    scale = 3.0  # Scale factor for visibility
    for i, (loading, stat_col) in enumerate(zip(loadings, STAT_COLS)):
        ax.arrow(0, 0, loading[0] * scale, loading[1] * scale,
                head_width=0.1, head_length=0.1, fc=TEXT_WHITE, ec=TEXT_WHITE, linewidth=1.5, alpha=0.8)
        ax.text(loading[0] * scale * 1.15, loading[1] * scale * 1.15,
               STAT_LABELS.get(stat_col, stat_col.upper()),
               fontsize=9, color=TEXT_WHITE, fontweight="bold", ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_DARK, edgecolor=TEXT_WHITE, alpha=0.8))
    
    # Label top 15 by debiased rank
    top15_debiased = debiased.nsmallest(15, "debiased_rank")
    for _, row in top15_debiased.iterrows():
        # Find index in stats dataframe
        stats_mask = stats["player_name"] == row["player_name"]
        if stats_mask.sum() > 0:
            stats_idx = stats[stats_mask].index[0]
            player_pos_in_stats = stats.index.get_loc(stats_idx)
            ax.annotate(row["player_name"],
                       (X_pca[player_pos_in_stats, 0], X_pca[player_pos_in_stats, 1]),
                       fontsize=8, color=TEXT_WHITE, alpha=0.9,
                       xytext=(5, 5), textcoords="offset points",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_DARK, edgecolor=get_position_color(row["position"]), alpha=0.8))
    
    ax.set_xlabel("PC1", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_ylabel("PC2", color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_title("PCA Biplot: PC1 vs PC2 with Loading Vectors", fontsize=TITLE_SIZE, fontweight="bold",
                 color=TEXT_WHITE, pad=20)
    add_position_legend(ax, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pca_biplot.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/pca_biplot.png")


# ============================================================================
# PLOT 11: Scatter Matrix
# ============================================================================

def plot_scatter_matrix():
    """Pairplot of key stats (ppg, apg, vorp, plus_minus, rpg) colored by position."""
    stats = pd.read_csv(STATS_PATH)
    
    key_stats = ["ppg", "apg", "vorp", "plus_minus", "rpg"]
    
    # Create custom palette list in order
    palette_list = [POSITION_COLORS["PG"], POSITION_COLORS["SG"], POSITION_COLORS["SF"], POSITION_COLORS["PF"], POSITION_COLORS["C"]]
    
    g = sns.pairplot(
        stats,
        vars=key_stats,
        hue="position",
        palette=palette_list,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15, "edgecolors": None},
        height=3.2,
    )
    
    g.fig.patch.set_facecolor(BG_DARK)
    
    # Style each subplot
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is not None:
                ax.set_facecolor(BG_PLOT)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color(TEXT_GRAY)
                ax.spines['left'].set_color(TEXT_GRAY)
                ax.tick_params(colors=TEXT_GRAY)
                ax.xaxis.label.set_color(TEXT_GRAY)
                ax.yaxis.label.set_color(TEXT_GRAY)
                for label in ax.get_xticklabels():
                    label.set_color(TEXT_GRAY)
                for label in ax.get_yticklabels():
                    label.set_color(TEXT_GRAY)
    
    g.fig.suptitle("Scatter Matrix: Key Stats by Position", fontsize=TITLE_SIZE, fontweight="bold",
                   color=TEXT_WHITE, y=0.995)
    
    # Style legend
    if hasattr(g, '_legend') and g._legend is not None:
        g._legend.get_frame().set_facecolor(BG_PLOT)
        g._legend.get_frame().set_edgecolor(TEXT_GRAY)
        for text in g._legend.get_texts():
            text.set_color(TEXT_GRAY)
    
    plt.tight_layout()
    g.savefig(os.path.join(PLOTS_DIR, "scatter_matrix.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print("Created: plots/scatter_matrix.png")


# ============================================================================
# PLOTS 12-23: Individual Stat Top 10 Charts
# ============================================================================

def plot_individual_stat_top10(stat_col, stat_label):
    """Generate top 10 chart for a single stat."""
    stats = pd.read_csv(STATS_PATH)
    ascending = (stat_col == "def_rating")
    if ascending:
        top10 = stats.nsmallest(10, stat_col).sort_values(stat_col)
    else:
        top10 = stats.nlargest(10, stat_col).sort_values(stat_col, ascending=False)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(right=0.88)
    setup_dark_style(fig, ax)
    
    n = len(top10)
    y_pos = np.arange(n)
    values = top10[stat_col].values
    v_max = values.max()
    v_min = values.min()
    # Leave room for rank left and value right
    x_min = v_min - 0.15 * (v_max - v_min) if v_max != v_min else v_min - 0.5
    x_max_plot = v_max * 1.25
    ax.set_xlim(x_min, x_max_plot)
    
    colors = [get_position_color(pos) for pos in top10["position"]]
    bars = ax.barh(y_pos, values, height=0.6, color=colors, alpha=0.8, edgecolor=TEXT_GRAY, linewidth=1)
    
    league_avg = stats[stat_col].mean()
    ax.axvline(league_avg, color=TEXT_GRAY, linestyle="--", linewidth=1.5, alpha=0.5, zorder=0)
    avg_label = f"{league_avg:.3f}" if stat_col.endswith("_pct") else f"{league_avg:.2f}"
    ax.text(league_avg * 1.02, n - 0.5, f"League Avg: {avg_label}",
            fontsize=9, color=TEXT_GRAY, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK, edgecolor=TEXT_GRAY, alpha=0.7))
    
    # Rank numbers: fixed left (data coords)
    rank_x = x_min + 0.02 * (v_max - v_min) if v_max != v_min else x_min + 0.02
    for i in range(n):
        ax.text(rank_x, i, f"#{i+1}", fontsize=16, fontweight="bold", color=GOLD, va="center", ha="left")
    
    # Player name and badge inside bar (2% from bar start = 0)
    name_start = 0.02 * v_max if v_max > 0 else 0.02
    for i, (idx, row) in enumerate(top10.iterrows()):
        ax.text(name_start, i, row["player_name"], fontsize=12, fontweight="bold",
                color=TEXT_WHITE, va="center", ha="left")
        pos_color = get_position_color(row["position"])
        badge_x = name_start + 0.05 * (v_max - v_min) if v_max != v_min else name_start + 0.5
        circle = Circle((badge_x, i), 0.14, color=pos_color, zorder=10)
        ax.add_patch(circle)
        ax.text(badge_x, i, row["position"], fontsize=8, fontweight="bold", color=BG_DARK,
                va="center", ha="center", zorder=11)
        # Stat value outside bar (right)
        val_str = f"{row[stat_col]:.3f}" if stat_col.endswith("_pct") else f"{row[stat_col]:.1f}"
        ax.text(v_max * 1.04, i, val_str, fontsize=13, fontweight="bold", color=TEXT_WHITE, va="center", ha="left")
    
    pos_counts = top10["position"].value_counts()
    accent_color = get_position_color(pos_counts.index[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel(stat_label, color=TEXT_GRAY, fontsize=LABEL_SIZE)
    ax.set_title(f"TOP 10 — {stat_label}", fontsize=TITLE_SIZE, fontweight="bold", color=TEXT_WHITE, pad=20)
    ax.axhline(y=-0.5, xmin=0.1, xmax=0.9, color=accent_color, linewidth=3, clip_on=False)
    add_position_legend(ax, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout(rect=[0.02, 0, 0.92, 0.98])
    filename = f"top10_{stat_col}.png"
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"Created: plots/{filename}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate all 23 plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING ALL PLOTS (Dark Theme)")
    print("=" * 70)
    
    # Main plots
    print("\n--- Main Plots ---")
    plot_top10_overall()
    plot_top5_mvp()
    plot_top5_dpoy()
    plot_all_nba_teams()
    plot_all_defensive_teams()
    plot_correlation_heatmap()
    plot_global_vs_position_rank()
    plot_kmeans_clusters()
    plot_position_boxplots()
    plot_pca_biplot()
    plot_scatter_matrix()
    
    # Individual stat charts
    print("\n--- Individual Stat Top 10 Charts ---")
    for stat_col in STAT_COLS:
        stat_label = STAT_LABELS.get(stat_col, stat_col.upper())
        plot_individual_stat_top10(stat_col, stat_label)
    
    # Verification checklist
    print("\n" + "=" * 70)
    print("PLOT GENERATION COMPLETE - VERIFICATION CHECKLIST")
    print("=" * 70)
    
    expected_plots = [
        "top10_overall.png",
        "top5_mvp.png",
        "top5_dpoy.png",
        "all_nba_teams.png",
        "all_defensive_teams.png",
        "correlation_heatmap.png",
        "global_vs_position_rank.png",
        "kmeans_clusters.png",
        "position_boxplots.png",
        "pca_biplot.png",
        "scatter_matrix.png",
    ] + [f"top10_{col}.png" for col in STAT_COLS]
    
    print(f"\nExpected: {len(expected_plots)} plots")
    print("\nFile Checklist:")
    for plot_file in expected_plots:
        filepath = os.path.join(PLOTS_DIR, plot_file)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  [OK] {plot_file:35s} ({size_kb:7.1f} KB)")
        else:
            print(f"  [MISSING] {plot_file}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
