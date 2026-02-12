"""
Generate a realistic synthetic NBA player stats dataset with correlated statistics.
Uses multivariate normal latent factors so good players tend to be good across categories.
"""

import os
import numpy as np
import pandas as pd
from numpy.random import default_rng

# Reproducibility
RNG = default_rng(42)
N_PLAYERS = 450

# Tier counts: superstar, starter, rotation, bench (scaled for 450, ~90 per position)
N_SUPERSTAR = 36
N_STARTER = 105
N_ROTATION = 165
N_BENCH = N_PLAYERS - N_SUPERSTAR - N_STARTER - N_ROTATION  # 144

POSITIONS = ["PG", "SG", "SF", "PF", "C"]

# Global first names (120+ unique, no real NBA player names)
FIRST_NAMES = [
    # American
    "James", "Michael", "David", "Daniel", "Anthony", "Marcus", "Tyler", "Brandon",
    "Justin", "Kyle", "Nathan", "Derek", "Caleb", "Ethan", "Ryan", "Aaron", "Trevor",
    "Connor", "Malik", "Andre", "Xavier", "Terrence", "Darius", "Jamal", "Isaiah",
    "Elijah", "Cameron", "Bryce", "Trey", "Jalen", "Grant", "Cole", "Blake", "Hayden",
    "Owen", "Landon", "Chase", "Hunter", "Cooper", "Parker", "Sawyer", "Bennett",
    "Weston", "Graham", "Brooks", "Reed", "Spencer", "Garrett", "Travis", "Wesley",
    # Latin American
    "Mateo", "Santiago", "Diego", "Luis", "Carlos", "Rafael", "Emilio", "Alejandro",
    "Gabriel", "Joaquin", "Fernando", "Ricardo", "Andres", "Pablo", "Sebastian",
    "Ignacio", "Rodrigo", "Miguel", "Eduardo", "Antonio", "Francisco", "Javier",
    # European
    "Luca", "Nikolas", "Stefan", "Henrik", "Theo", "Bastian", "Emil", "Kristian",
    "Sven", "Florian", "Matteo", "Lorenzo", "Hugo", "Adrien", "Yannick", "Dominik",
    "Felix", "Leon", "Max", "Finn", "Erik", "Lars", "Oscar", "Viktor", "Anton",
    # African
    "Kwame", "Ousmane", "Amadou", "Chidi", "Kofi", "Jelani", "Sekou", "Zuberi",
    "Tendai", "Kato", "Emeka", "Idris", "Baraka", "Jabari", "Keenan", "Rashad",
    # Asian/Pacific
    "Kai", "Ryo", "Jin", "Taro", "Sho", "Min", "Hiro", "Koa", "Keanu", "Manu", "Tane",
    "Kenji", "Yuki", "Ren", "Haruto", "Asahi", "Sota", "Daisuke",
    # Middle Eastern
    "Omar", "Tariq", "Amir", "Rashid", "Samir", "Karim", "Zaid", "Hamza",
    "Khalil", "Youssef", "Nasser", "Faris", "Rami", "Tariq",
    # Unique / less common
    "Caspian", "Zephyr", "Orion", "Thiago", "Eero", "Stellan", "Nico", "Callum",
    "Ronan", "Beckett", "Ansel", "Ender", "Arlo", "Bodhi", "Cruz", "Dax", "Ezra",
    "Griffin", "Holden", "Knox", "Lennox", "Milo", "Nico", "Phoenix", "Quinn",
    "River", "Silas", "Titus", "Vance", "Zane",
]

# Global last names (120+ unique, no real NBA player last names)
LAST_NAMES = [
    # American
    "Mitchell", "Parker", "Bennett", "Collins", "Rivera", "Hayes", "Sullivan", "Porter",
    "Foster", "Brooks", "Coleman", "Reed", "Morgan", "Cooper", "Bell", "Ward", "Torres",
    "Reeves", "Hartley", "Walsh", "Mercer", "Donovan", "Gallagher", "Whitfield", "Langston",
    "Ashworth", "Kensington", "Blackwell", "Creed", "Voss", "Thorn", "Caine", "Wren",
    "Prescott", "Hale", "Greyson", "Lennox", "Crane", "Aldridge", "Marek", "Sutton",
    "Channing", "Ellis", "Holloway", "Kendrick", "Lancaster", "Marlow", "Nash", "Quinn",
    # Latin American
    "Morales", "Gutierrez", "Castillo", "Delgado", "Romero", "Salazar", "Mendoza", "Aguilar",
    "Fuentes", "Escobar", "Navarro", "Cordova", "Paredes", "Villareal", "Contreras",
    "Beltran", "Cervantes", "Espinoza", "Herrera", "Luna", "Marquez", "Rios", "Vargas",
    # European
    "Lindqvist", "Bergmann", "Novak", "Kowalski", "Dimitrov", "Ferreira", "Janssen", "Eriksson",
    "Brandt", "Schreiber", "Castellano", "Petrova", "Andersen", "Moller", "Visser",
    "Hoffmann", "Kozlov", "Nielsen", "Vasquez", "Wagner", "Zimmermann",
    # African
    "Okafor", "Diallo", "Mensah", "Afolabi", "Nkrumah", "Mwangi", "Diop", "Toure",
    "Bankole", "Okeke", "Abara", "Sesay", "Kamara", "Osei", "Boateng",
    # Asian/Pacific
    "Nakamura", "Tanaka", "Chen", "Zhao", "Patel", "Sharma", "Watanabe", "Takahashi",
    "Hayashi", "Kapoor", "Singh", "Liang", "Kim", "Park", "Yamamoto", "Suzuki",
    # Middle Eastern
    "Al-Rashid", "Hadid", "Farouk", "Nassir", "Khalil", "Sabbagh", "Mansouri", "Darwish",
    "Haddad", "Malouf", "Qureshi",
]


def generate_names(n: int) -> list[str]:
    """Generate unique full names by combining first + last; re-roll on duplicate until unique."""
    names = set()
    while len(names) < n:
        first = RNG.choice(FIRST_NAMES)
        last = RNG.choice(LAST_NAMES)
        full = f"{first} {last}"
        names.add(full)
    return list(names)


def assign_positions(n: int) -> np.ndarray:
    """Assign positions roughly evenly (~90 per position for 450)."""
    positions = np.array(POSITIONS * (n // len(POSITIONS)))
    remainder = n - len(positions)
    if remainder > 0:
        positions = np.concatenate([positions, RNG.choice(POSITIONS, size=remainder, replace=True)])
    RNG.shuffle(positions)
    return positions


def draw_latent_skills(n: int) -> np.ndarray:
    """
    Draw (overall_quality, offensive_skill, defensive_skill) from correlated MVN.
    Good players tend to be good on both ends but with some variation.
    """
    mean = np.zeros(3)
    # Positive correlations: overall with both; offensive/defensive somewhat correlated
    cov = np.array([
        [1.0, 0.65, 0.60],
        [0.65, 1.0, 0.45],
        [0.60, 0.45, 1.0],
    ])
    return RNG.multivariate_normal(mean, cov, size=n)


def assign_tiers(latent_overall: np.ndarray) -> np.ndarray:
    """Assign tier by ranking overall quality: superstar, starter, rotation, bench."""
    ranks = np.argsort(np.argsort(-latent_overall))  # 0 = best
    tier = np.zeros(N_PLAYERS, dtype=int)
    tier[ranks < N_SUPERSTAR] = 0
    tier[(ranks >= N_SUPERSTAR) & (ranks < N_SUPERSTAR + N_STARTER)] = 1
    tier[(ranks >= N_SUPERSTAR + N_STARTER) & (ranks < N_SUPERSTAR + N_STARTER + N_ROTATION)] = 2
    tier[ranks >= N_SUPERSTAR + N_STARTER + N_ROTATION] = 3
    return tier


def position_index(pos: np.ndarray) -> np.ndarray:
    """Map position string to index 0..4 for PG..C."""
    return np.array([POSITIONS.index(p) for p in pos])


def generate_ppg(position: np.ndarray, tier: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """
    PPG using log-normal so natural max lands around 33-34.5.
    Shape: most 6-15, starters 15-22, stars 22-28, elite 28-32, absolute max 33-34.5.
    Nobody above 35; only 1-2 outliers in 33-35 range.
    """
    # Base by position (slightly lower to hit league avg 10-14)
    base = np.array([11.5, 12, 11.5, 10.5, 9.5])[position_index(position)]
    tier_bonus = np.array([8, 3, 0, -2.5])[tier]  # superstar +8, starter +3, rotation 0, bench -2.5
    quality_factor = 0.8 * latent[:, 0] + 0.6 * latent[:, 1]
    # Log-normal: small sigma so tail ends naturally near 33-34
    log_mean = np.log(np.maximum(base + tier_bonus + quality_factor + RNG.normal(0, 0.6, N_PLAYERS), 4.0))
    log_std = 0.30  # Keeps 99th percentile around 32-34
    log_ppg = RNG.normal(log_mean, log_std)
    ppg = np.exp(log_ppg)
    # Re-sample any below 3
    too_low = ppg < 3
    if too_low.any():
        log_mean_fix = np.log(np.maximum(base[too_low] + tier_bonus[too_low] + 4.0, 4.0))
        ppg[too_low] = np.exp(RNG.normal(log_mean_fix, 0.25))
    # Cap: nobody above 35; anyone above 34.5 becomes 33-34.5 (1-2 elite scorers)
    above = ppg > 34.5
    if above.any():
        n_above = above.sum()
        ppg[above] = RNG.uniform(33.0, 34.5, size=n_above)
    ppg = np.minimum(ppg, 35.0)  # Hard cap 35
    return np.round(ppg, 1)


def generate_rpg(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Rebounds: wider normal distributions, no clipping."""
    # C: mean 9.5, std 2.8; PF: mean 7.0, std 2.2; SF: mean 5.2, std 1.5; SG: mean 3.5, std 1.2; PG: mean 3.8, std 1.4
    means = np.array([3.8, 3.5, 5.2, 7.0, 9.5])[position_index(position)]
    stds = np.array([1.4, 1.2, 1.5, 2.2, 2.8])[position_index(position)]
    rpg = means + 0.4 * latent[:, 0] * stds + RNG.normal(0, stds)
    rpg = np.maximum(rpg, 0.5)  # Floor only, no ceiling
    rpg = np.round(rpg, 1)
    # Ensure no negatives
    rpg = np.maximum(rpg, 0.5)
    return rpg


def generate_apg(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Assists: wider normal distributions, no clipping."""
    # PG: mean 6.5, std 2.5; SG: mean 3.2, std 1.5; SF: mean 2.5, std 1.2; PF: mean 2.0, std 1.0; C: mean 2.2, std 1.5
    means = np.array([6.5, 3.2, 2.5, 2.0, 2.2])[position_index(position)]
    stds = np.array([2.5, 1.5, 1.2, 1.0, 1.5])[position_index(position)]
    apg = means + 0.5 * latent[:, 1] * stds + RNG.normal(0, stds)
    apg = np.maximum(apg, 0.3)  # Floor only
    apg = np.round(apg, 1)
    # Ensure no negatives
    apg = np.maximum(apg, 0.3)
    return apg


def generate_bpg(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Blocks: wider normal distributions, minimum floor 0.1, no ceiling."""
    # C: mean 1.8, std 0.9; PF: mean 1.0, std 0.6; SF: mean 0.7, std 0.5; SG: mean 0.4, std 0.25; PG: mean 0.3, std 0.2
    means = np.array([0.3, 0.4, 0.7, 1.0, 1.8])[position_index(position)]
    stds = np.array([0.2, 0.25, 0.5, 0.6, 0.9])[position_index(position)]
    bpg = means + 0.4 * latent[:, 2] * stds + RNG.normal(0, stds)
    bpg = np.maximum(bpg, 0.1)  # Floor only
    return np.round(bpg, 1)


def generate_spg(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Steals: wider normal distributions, minimum floor 0.2, no ceiling."""
    # PG: mean 1.3, std 0.5; SG: mean 1.1, std 0.45; SF: mean 0.9, std 0.4; PF: mean 0.7, std 0.3; C: mean 0.6, std 0.3
    means = np.array([1.3, 1.1, 0.9, 0.7, 0.6])[position_index(position)]
    stds = np.array([0.5, 0.45, 0.4, 0.3, 0.3])[position_index(position)]
    spg = means + 0.3 * latent[:, 2] * stds + 0.2 * latent[:, 1] * stds + RNG.normal(0, stds)
    spg = np.maximum(spg, 0.2)  # Floor only
    spg = np.round(spg, 1)
    # Ensure no negatives
    spg = np.maximum(spg, 0.2)
    return spg


def generate_plus_minus(latent: np.ndarray) -> np.ndarray:
    """Plus/minus: normal distribution, mean 0, std 4.5, correlated with quality."""
    quality_bonus = 2.0 * latent[:, 0] + 0.6 * latent[:, 1] + 0.6 * latent[:, 2]
    plus_minus = quality_bonus + RNG.normal(0, 4.5, N_PLAYERS)
    return np.round(plus_minus, 1)


def generate_def_rating(latent: np.ndarray) -> np.ndarray:
    """
    Defensive rating: mean 110, std 5.5. Lower is better.
    Good: 96-104, Average: 104-112, Bad: 112-120, Terrible: 120-125.
    """
    def_rating = 110 - 4.0 * latent[:, 2] - 1.5 * latent[:, 0] + RNG.normal(0, 5.5, N_PLAYERS)
    return np.round(def_rating, 1)


def generate_off_rating(latent: np.ndarray) -> np.ndarray:
    """
    Offensive rating: mean 109, std 4.5. Real elite is 118-125.
    Natural range ~96-124. If >126, re-sample from normal(118, 2.5).
    """
    off_rating = 109 + 3.0 * latent[:, 1] + 1.5 * latent[:, 0] + RNG.normal(0, 4.5, N_PLAYERS)
    # Re-sample any exceeding 126 so top 10 land in 119-124
    too_high = off_rating > 126
    if too_high.any():
        off_rating[too_high] = np.round(RNG.normal(118, 2.5, size=too_high.sum()), 1)
    off_rating = np.minimum(off_rating, 126)  # Cap at 126
    return np.round(off_rating, 1)


def generate_fg_pct(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """FG%: position-based normal distributions, floor 0.32, no ceiling."""
    # C: mean 0.54, std 0.05; PF: mean 0.49, std 0.045; SF: mean 0.46, std 0.04; SG: mean 0.44, std 0.04; PG: mean 0.43, std 0.04
    means = np.array([0.43, 0.44, 0.46, 0.49, 0.54])[position_index(position)]
    stds = np.array([0.04, 0.04, 0.04, 0.045, 0.05])[position_index(position)]
    fg_pct = means + 0.02 * latent[:, 1] * stds + RNG.normal(0, stds)
    fg_pct = np.maximum(fg_pct, 0.32)  # Floor only
    return np.round(fg_pct, 3)


def generate_three_pt_pct(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """3P%: Guards/SF mean 0.36, std 0.04; PF mean 0.34, std 0.045; C mean 0.28, std 0.06. Floor 0.18."""
    # Guards/SF: mean 0.36, std 0.04; PF: mean 0.34, std 0.045; C: mean 0.28, std 0.06
    means = np.array([0.36, 0.36, 0.36, 0.34, 0.28])[position_index(position)]
    stds = np.array([0.04, 0.04, 0.04, 0.045, 0.06])[position_index(position)]
    three_pt_pct = means + 0.015 * latent[:, 1] * stds + RNG.normal(0, stds)
    three_pt_pct = np.maximum(three_pt_pct, 0.18)  # Floor only
    return np.round(three_pt_pct, 3)


def generate_ft_pct(position: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """FT%: PG/SG mean 0.82, std 0.06; SF mean 0.78, std 0.07; PF mean 0.74, std 0.08; C mean 0.68, std 0.10. Floor 0.40."""
    means = np.array([0.82, 0.82, 0.78, 0.74, 0.68])[position_index(position)]
    stds = np.array([0.06, 0.06, 0.07, 0.08, 0.10])[position_index(position)]
    ft_pct = means + 0.015 * latent[:, 1] * stds + RNG.normal(0, stds)
    ft_pct = np.maximum(ft_pct, 0.40)  # Floor only
    return np.round(ft_pct, 3)


def generate_vorp(latent: np.ndarray) -> np.ndarray:
    """VORP: normal distribution mean 0.8, std 2.0, correlated with quality, no clipping."""
    vorp = 0.8 + 1.0 * latent[:, 0] + 0.4 * latent[:, 1] + 0.25 * latent[:, 2] + RNG.normal(0, 2.0, N_PLAYERS)
    return np.round(vorp, 1)


def enforce_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all stat values are unique by adding tiny random noise to duplicates.
    For counting stats: ±0.1-0.4; for percentages: ±0.001-0.008.
    Iterates until all values are unique (up to 20 iterations).
    """
    df = df.copy()
    counting_stats = ["ppg", "rpg", "apg", "bpg", "spg", "plus_minus", "def_rating", "off_rating", "vorp"]
    percentage_stats = ["fg_pct", "three_pt_pct", "ft_pct"]
    
    max_iterations = 20
    for iteration in range(max_iterations):
        all_unique = True
        for col in counting_stats + percentage_stats:
            if col not in df.columns:
                continue
            values = df[col].values.copy()
            rounded = np.round(values, 1 if col in counting_stats else 3)
            
            # Find duplicates
            unique_vals, counts = np.unique(rounded, return_counts=True)
            duplicates = unique_vals[counts > 1]
            
            if len(duplicates) > 0:
                all_unique = False
                for dup_val in duplicates:
                    mask = np.abs(rounded - dup_val) < 0.0001  # Find all matching this value
                    indices = np.where(mask)[0]
                    # Keep first occurrence, add noise to rest
                    for idx in indices[1:]:
                        if col in counting_stats:
                            # More aggressive noise for counting stats
                            noise = RNG.uniform(0.1, 0.4) * (1 if RNG.random() < 0.5 else -1)
                            new_val = values[idx] + noise
                            # Respect floors for certain stats
                            if col == "rpg":
                                new_val = max(new_val, 0.5)
                            elif col == "apg":
                                new_val = max(new_val, 0.3)
                            elif col == "bpg":
                                new_val = max(new_val, 0.1)
                            elif col == "spg":
                                new_val = max(new_val, 0.2)
                            values[idx] = np.round(new_val, 1)
                        else:
                            # More aggressive noise for percentages
                            noise = RNG.uniform(0.001, 0.008) * (1 if RNG.random() < 0.5 else -1)
                            new_val = values[idx] + noise
                            # Respect floors/ceilings (match apply_percentage_bounds)
                            if col == "fg_pct":
                                new_val = max(0.34, min(new_val, 0.735))
                            elif col == "three_pt_pct":
                                new_val = max(0.18, min(new_val, 0.487))
                            elif col == "ft_pct":
                                new_val = max(0.42, min(new_val, 0.96))
                            values[idx] = np.round(new_val, 3)
                df[col] = values
        
        if all_unique:
            break
    
    return df


def apply_percentage_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    After ALL generation and outlier injection: apply hard floor/ceiling for percentage stats.
    FT%: 42%–96% (impossible to exceed 100%). FG%: 34%–70%. 3P%: 18%–48%.
    """
    df = df.copy()
    # FT%: the one stat where ceiling is literal
    df["ft_pct"] = np.clip(df["ft_pct"], 0.420, 0.960)
    df["ft_pct"] = np.round(df["ft_pct"], 3)
    # FG%: sanity bounds
    df["fg_pct"] = np.clip(df["fg_pct"], 0.340, 0.735)
    df["fg_pct"] = np.round(df["fg_pct"], 3)
    # 3P%: sanity bounds
    df["three_pt_pct"] = np.clip(df["three_pt_pct"], 0.180, 0.487)
    df["three_pt_pct"] = np.round(df["three_pt_pct"], 3)
    return df


def inject_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject memorable statistical outliers / archetypes after base generation.
    Also ensures values remain within realistic NBA ranges.
    """
    df = df.copy()

    # Add archetype column if missing (object dtype so we can store strings)
    if "archetype" not in df.columns:
        df["archetype"] = pd.Series([np.nan] * len(df), dtype="object")

    # Slight league-wide scoring taper (optional; base distribution already targets 10-14)
    # df["ppg"] unchanged; caps applied in generate_ppg and below per-archetype

    chosen: set[int] = set()

    def available_indices(mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask)[0]
        idx = np.array([i for i in idx if i not in chosen], dtype=int)
        return idx

    def choose(mask: np.ndarray, n_min: int, n_max: int) -> np.ndarray:
        n = int(RNG.integers(n_min, n_max + 1))
        idx = available_indices(mask)
        if len(idx) == 0:
            return np.array([], dtype=int)
        n = min(n, len(idx))
        picked = RNG.choice(idx, size=n, replace=False)
        chosen.update(picked.tolist())
        return picked

    # Memorable names for outliers (no real NBA player names)
    SPECIAL_NAMES = [
        "Zeus Crane", "Dante Bridges", "Koa Nakamura", "Nova Reyes", "Atlas King",
        "Orion Carter", "Titan Rivers", "Echo Zhang", "Phoenix Hale", "Zaire Storm",
        "Rio Valdez", "Cael Winters", "Miles Orion", "Jett Ashworth", "Roan Sato",
        "Luca Vega", "Seneca Rhodes", "Arlo Knight", "Ares Collins", "Jaxson Steele",
        "Caspian Creed", "Zephyr Thorn", "Stellan Voss", "Eero Lindqvist", "Ronan Hale",
        "Beckett Mercer", "Idris Diallo", "Ansel Whitfield", "Ender Navarro", "Thiago Castillo",
        "Nico Castellano", "Callum Donovan", "Griffin Blackwell", "Phoenix Kensington",
    ]
    name_pool = list(SPECIAL_NAMES)
    RNG.shuffle(name_pool)

    def take_name() -> str:
        if name_pool:
            return name_pool.pop()
        return None

    # 1. Point Center (4-5 players): C with 7-11 apg, rpg 9+, lower bpg 1-2.
    mask_c = df["position"] == "C"
    idx_pc = choose(mask_c, 4, 5)
    for i in idx_pc:
        df.at[i, "archetype"] = "Point Center"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "apg"] = np.round(RNG.uniform(7.0, 11.0), 1)
        df.at[i, "rpg"] = np.round(RNG.uniform(9.0, 13.0), 1)
        df.at[i, "bpg"] = np.round(RNG.uniform(1.0, 2.0), 1)
        df.at[i, "ppg"] = np.round(min(RNG.uniform(16.0, 24.0), 32.0), 1)

    # 2. Rebounding Guard (4-5 players): PG/SG with 8-12 rpg, apg 5+, ppg 16-25.
    mask_rg = df["position"].isin(["PG", "SG"])
    idx_rg = choose(mask_rg, 4, 5)
    for i in idx_rg:
        df.at[i, "archetype"] = "Rebounding Guard"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "rpg"] = np.round(RNG.uniform(8.0, 12.0), 1)
        df.at[i, "apg"] = np.round(RNG.uniform(5.0, 9.0), 1)
        df.at[i, "ppg"] = np.round(min(RNG.uniform(16.0, 25.0), 32.0), 1)

    # 3. Shot-blocking Wing (3-4 players): SF/SG with 2.5-4.0 bpg, rpg 6-9.
    mask_wing = df["position"].isin(["SF", "SG"])
    idx_sbw = choose(mask_wing, 3, 4)
    for i in idx_sbw:
        df.at[i, "archetype"] = "Shot-blocking Wing"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "bpg"] = np.round(RNG.uniform(2.5, 4.0), 1)
        df.at[i, "rpg"] = np.round(RNG.uniform(6.0, 9.0), 1)
        # Slight defensive bump
        df.at[i, "def_rating"] = np.round(RNG.uniform(96.0, 103.0), 1)

    # 4. Scoring Big with Range (4-5 players): C/PF, 3P% 0.38-0.43, ppg 20+, rpg 6-8.
    mask_big_range = df["position"].isin(["C", "PF"])
    idx_sbr = choose(mask_big_range, 4, 5)
    for i in idx_sbr:
        df.at[i, "archetype"] = "Scoring Big with Range"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "three_pt_pct"] = np.round(RNG.uniform(0.38, 0.43), 3)
        df.at[i, "ppg"] = np.round(min(RNG.uniform(20.0, 28.0), 32.0), 1)
        df.at[i, "rpg"] = np.round(RNG.uniform(6.0, 8.0), 1)
        df.at[i, "fg_pct"] = np.round(RNG.uniform(0.48, 0.56), 3)

    # 5. Defensive Guard Menace (4-5 players): PG/SG with spg 2.2-3.0, def_rating 96-101, ppg 10-16.
    mask_dg = df["position"].isin(["PG", "SG"])
    idx_dg = choose(mask_dg, 4, 5)
    for i in idx_dg:
        df.at[i, "archetype"] = "Defensive Guard Menace"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "spg"] = np.round(RNG.uniform(2.2, 3.0), 1)
        df.at[i, "def_rating"] = np.round(RNG.uniform(96.0, 101.0), 1)
        df.at[i, "ppg"] = np.round(RNG.uniform(10.0, 16.0), 1)

    # 6. The Empty Stats Guy (4-5 players): ppg 20-26 max 32, terrible plus_minus, def_rating 116-125, low fg_pct, negative vorp.
    mask_any = np.ones(len(df), dtype=bool)
    idx_esg = choose(mask_any, 4, 5)
    for i in idx_esg:
        df.at[i, "archetype"] = "Empty Stats Guy"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "ppg"] = np.round(min(RNG.uniform(20.0, 26.0), 32.0), 1)
        df.at[i, "plus_minus"] = np.round(RNG.uniform(-8.0, -4.0), 1)
        df.at[i, "def_rating"] = np.round(RNG.uniform(116.0, 125.0), 1)
        df.at[i, "fg_pct"] = np.round(RNG.uniform(0.39, 0.42), 3)
        df.at[i, "vorp"] = np.round(RNG.uniform(-1.0, -0.1), 1)

    # 7. The Quiet MVP (3-4 players): ppg 16-22 (cap 32), elite plus_minus, def_rating, vorp, fg%.
    idx_qm = choose(mask_any, 3, 4)
    for i in idx_qm:
        df.at[i, "archetype"] = "Quiet MVP"
        new_name = take_name()
        if new_name:
            df.at[i, "player_name"] = new_name
        df.at[i, "ppg"] = np.round(min(RNG.uniform(16.0, 22.0), 32.0), 1)
        df.at[i, "plus_minus"] = np.round(RNG.uniform(8.0, 12.0), 1)
        df.at[i, "def_rating"] = np.round(RNG.uniform(95.0, 100.0), 1)
        df.at[i, "vorp"] = np.round(RNG.uniform(4.5, 6.0), 1)
        df.at[i, "fg_pct"] = np.round(RNG.uniform(0.52, 0.62), 3)

    # No final clipping - let distributions determine ranges naturally

    outliers = df[df["archetype"].notna()].copy()
    # Sort by archetype for readability
    outliers = outliers.sort_values(["archetype", "ppg"], ascending=[True, False])
    return df, outliers


def main():
    # Names and positions
    player_name = generate_names(N_PLAYERS)
    position = assign_positions(N_PLAYERS)

    # Correlated latent skills (multivariate normal)
    latent = draw_latent_skills(N_PLAYERS)
    tier = assign_tiers(latent[:, 0])

    # Generate all stats (each uses latent + position where relevant + noise)
    df = pd.DataFrame({
        "player_name": player_name,
        "position": position,
        "ppg": generate_ppg(position, tier, latent),
        "rpg": generate_rpg(position, latent),
        "apg": generate_apg(position, latent),
        "bpg": generate_bpg(position, latent),
        "spg": generate_spg(position, latent),
        "plus_minus": generate_plus_minus(latent),
        "def_rating": generate_def_rating(latent),
        "off_rating": generate_off_rating(latent),
        "fg_pct": generate_fg_pct(position, latent),
        "three_pt_pct": generate_three_pt_pct(position, latent),
        "ft_pct": generate_ft_pct(position, latent),
        "vorp": generate_vorp(latent),
    })

    # Inject archetypes / outliers and apply final scoring adjustments
    df, outliers = inject_outliers(df)

    # Apply percentage bounds (FT% 42–96%, FG% 34–73.5%, 3P% 18–48.7%)
    df = apply_percentage_bounds(df)
    
    # Enforce uniqueness of stat values
    df = enforce_uniqueness(df)

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)
    out_path = "data/nba_synthetic_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}\n")

    # Summary
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal player count: {len(df)}")
    print(f"Columns: {df.shape[1]}\n")

    print("League average (every stat):")
    numeric = df.select_dtypes(include=[np.number])
    avgs = numeric.mean()
    for col in numeric.columns:
        v = avgs[col]
        if col.endswith("_pct"):
            print(f"  {col}: {v:.3f}")
        else:
            print(f"  {col}: {v:.2f}")
    print()

    # All outlier players with their archetypes
    if not outliers.empty:
        print("=" * 70)
        print("ALL OUTLIER PLAYERS (archetype + key stats)")
        print("=" * 70)
        cols = [
            "player_name", "position", "archetype",
            "ppg", "rpg", "apg", "bpg", "spg",
            "plus_minus", "def_rating", "vorp",
        ]
        # Only include cols that exist
        cols = [c for c in cols if c in outliers.columns]
        print(outliers[cols].to_string(index=False))
        print(f"\nTotal outliers: {len(outliers)}")
    else:
        print("No outlier archetypes were injected.")

    # Verification: top/bottom 10, std dev, uniqueness, duplicates
    print("\n" + "=" * 70)
    print("VERIFICATION: STAT DISTRIBUTION AND UNIQUENESS")
    print("=" * 70)
    
    stat_cols = ["ppg", "rpg", "apg", "bpg", "spg", "plus_minus", "def_rating", "off_rating", 
                 "fg_pct", "three_pt_pct", "ft_pct", "vorp"]
    
    for col in stat_cols:
        if col not in df.columns:
            continue
        values = df[col].values
        rounded = np.round(values, 1 if col not in ["fg_pct", "three_pt_pct", "ft_pct"] else 3)
        unique_count = len(np.unique(rounded))
        std_val = np.std(values)
        
        # Find duplicates
        unique_vals, counts = np.unique(rounded, return_counts=True)
        duplicates = unique_vals[counts > 1]
        dup_info = []
        if len(duplicates) > 0:
            for dup_val in duplicates[:3]:  # Show first 3 duplicate values with player names
                dup_indices = np.where(np.abs(rounded - dup_val) < 0.0001)[0]
                player_names = [df.iloc[i]['player_name'] for i in dup_indices[:3]]  # First 3 names
                dup_info.append(f"{dup_val} ({len(dup_indices)} players: {', '.join(player_names)}{'...' if len(dup_indices) > 3 else ''})")
        
        # Top and bottom
        top10_idx = np.argsort(values)[-10:][::-1]
        bottom10_idx = np.argsort(values)[:10]
        
        print(f"\n{col.upper()}:")
        print(f"  Std Dev: {std_val:.3f}")
        print(f"  Unique values: {unique_count}/450")
        if dup_info:
            print(f"  WARNING: DUPLICATES FOUND:")
            for info in dup_info:
                print(f"    - {info}")
        else:
            print(f"  OK: No duplicates")
        print(f"  Highest: {df.iloc[top10_idx[0]]['player_name']} ({values[top10_idx[0]]:.3f})")
        print(f"  Lowest: {df.iloc[bottom10_idx[0]]['player_name']} ({values[bottom10_idx[0]]:.3f})")
        print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("Descriptive statistics (numeric):")
    print(numeric.describe().to_string())

    # --- Verification: stat fixes (PPG, Off/Def Rtg, FT%, FG%, 3P%) ---
    print("\n" + "=" * 70)
    print("VERIFICATION: STAT BOUNDS (PPG, RATINGS, PERCENTAGES)")
    print("=" * 70)

    # PPG: min, max, mean, std, top 5 scorers
    ppg = df["ppg"]
    print("\nPPG: min={:.1f}, max={:.1f}, mean={:.2f}, std={:.2f}".format(ppg.min(), ppg.max(), ppg.mean(), ppg.std()))
    top5_ppg = df.nlargest(5, "ppg")[["player_name", "position", "ppg"]]
    print("Top 5 scorers:")
    print(top5_ppg.to_string(index=False))

    # Off Rating: min, max, mean, top 5
    off = df["off_rating"]
    print("\nOff Rating: min={:.1f}, max={:.1f}, mean={:.2f}".format(off.min(), off.max(), off.mean()))
    top5_off = df.nlargest(5, "off_rating")[["player_name", "position", "off_rating"]]
    print("Top 5 off rating:")
    print(top5_off.to_string(index=False))

    # Def Rating: min, max, mean; bottom 5 (best), top 5 (worst)
    def_r = df["def_rating"]
    print("\nDef Rating: min={:.1f}, max={:.1f}, mean={:.2f} (lower = better)".format(def_r.min(), def_r.max(), def_r.mean()))
    best5_def = df.nsmallest(5, "def_rating")[["player_name", "position", "def_rating"]]
    worst5_def = df.nlargest(5, "def_rating")[["player_name", "position", "def_rating"]]
    print("Bottom 5 (best defenders):")
    print(best5_def.to_string(index=False))
    print("Top 5 (worst defenders):")
    print(worst5_def.to_string(index=False))

    # FT%: min, max, confirm 42–96%
    ft = df["ft_pct"]
    print("\nFT%: min={:.3f} ({:.1f}%), max={:.3f} ({:.1f}%)".format(ft.min(), ft.min() * 100, ft.max(), ft.max() * 100))
    ok_ft = (ft >= 0.42).all() and (ft <= 0.96).all()
    print("  Within 42-96%: {}".format("YES" if ok_ft else "NO"))

    # FG%: min, max, confirm range
    fg = df["fg_pct"]
    print("\nFG%: min={:.3f}, max={:.3f}".format(fg.min(), fg.max()))
    ok_fg = (fg >= 0.34).all() and (fg <= 0.735).all()
    print("  Within 34-73.5%: {}".format("YES" if ok_fg else "NO"))

    # 3P%: min, max, confirm range
    threes = df["three_pt_pct"]
    print("\n3P%: min={:.3f}, max={:.3f}".format(threes.min(), threes.max()))
    ok_3p = (threes >= 0.18).all() and (threes <= 0.487).all()
    print("  Within 18-48.7%: {}".format("YES" if ok_3p else "NO"))

    print("\nDone.")


if __name__ == "__main__":
    main()
