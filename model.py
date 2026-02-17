"""PGA Best Ball Draft Model (extracted from DRAFT_GUI.py)

This module contains ONLY data loading + model logic (no Tkinter).
It is safe to import from a desktop GUI (Tkinter) or a web backend (Flask/FastAPI).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION (MATCHES MODEL)
# ============================================================

TEAM_SIZE = 12
MIN_PLAYERS_PER_EVENT = 6

DEFAULT_ALPHA = 0.5
LAMBDA_PER_ROUND = 0.1
TOP_N_RECOMMENDATIONS = 10

# Add the Albatross-specific constants at the top with other configs
# When marginal best-ball gains are tiny, switch to "raw projection" ranking (Albatross only)
MARGINAL_SWITCH_EPS = 5.0
FALLBACK_PENALTY_WEIGHT = 1.0
FALLBACK_BONUS_WEIGHT = 1.0

# ------------------------------------------------------------
# CACHED STATE
# ------------------------------------------------------------
# These are set by init_model(). Keeping them module-level makes it
# easy to reuse the model from both a GUI and a web backend.
df = None  # type: ignore
event_cols = []

# Cached event weights (recomputed only when alpha changes)
EVENT_WEIGHTS = None
EVENT_WEIGHTS_ALPHA = None

URGENCY_MULTIPLIER = {
    "urgent": 2.0,
    "soon": 1.0
}

CONTEST_FORMATS = {
    "The Scramble": {
        "Sony": "Round1",
        "Amex": "Round1",
        "Farmers": "Round1",
        "Phoenix": "Round1",
        "Pebble": "Round1",
        "Genesis": "Round1",
        "Cognizant": "Round1",
        "API": "Round2",
        "Players": "Round2",
        "Valspar": "Round2",
        "Houston": "Round2",
        "Valero": "Round2",
        "Masters": "Round2",
        "Heritage": "Round2",
        "Miami": "Round3",
        "Truist": "Round3",
        "PGA": "Round3",
        "CJ Cup": "Round3",
        "Schwab": "Round3",
        "Memorial": "Round3",
        "Canadian": "Round4",
        "US Open": "Round4",
        "Travelers": "Round4",
        "Deere": "Round4",
        "Scottish": "Round4",
        "Open": "Round4",
    },
    "The Albatross": {
        "Masters": "Round1",
        "PGA": "Round2",
        "US Open": "Round3",
        "Open": "Round4",
    }
}

# Replace the old EVENT_TO_ROUND with a function that gets the current contest
EVENT_TO_ROUND = CONTEST_FORMATS["The Scramble"]  # Default

ROUND_MULTIPLIERS = {
    "Round1": 1.15,     # Default: 1.15
    "Round2": 1.05,     # Default: 1.05
    "Round3": 1.00,     # Default: 1.00
    "Round4": 0.95,     # Default: 0.95
}

# Module-level model state (initialized by init_model)

df: pd.DataFrame | None = None

event_cols: list[str] | None = None

EVENT_WEIGHTS = None  # cached event weights

# Fast cached arrays (set in init_model)
PLAYER_NAMES = None          # list[str]
NAME_TO_I = None             # dict[str, int]
M = None                     # np.ndarray (P, E) float with NaN
ADP_ARR = None               # np.ndarray (P,) float
EVENT_MULT_ARR = None        # np.ndarray (E,) float
EVENT_W_ARR = None           # np.ndarray (E,) float aligned to event_cols

def init_model(csv_path: str, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Load the CSV, clean numeric event columns, set module globals, and cache event weights."""
    global df, event_cols, EVENT_WEIGHTS, EVENT_WEIGHTS_ALPHA
    _df = pd.read_csv(csv_path)
    # Clean ADP (ensure numeric sorting & consistent display)
    if "ADP" in _df.columns:
        _df["ADP"] = (
            _df["ADP"]
            .replace("-", np.nan)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(999.0)
        )

    # Determine which event columns are present in the CSV
    _event_cols = [c for c in EVENT_TO_ROUND.keys() if c in _df.columns]
    # Clean event columns (match original GUI/model behavior)
    _df[_event_cols] = (
        _df[_event_cols]
        .replace('-', np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )
    _df.set_index('Name', inplace=True)
    df = _df
    event_cols = _event_cols
    EVENT_WEIGHTS = compute_event_weights(df, event_cols, DEFAULT_ALPHA)

    # ---- FAST CACHES ----
    global PLAYER_NAMES, NAME_TO_I, M, ADP_ARR, EVENT_MULT_ARR, EVENT_W_ARR

    PLAYER_NAMES = df.index.tolist()
    NAME_TO_I = {n: i for i, n in enumerate(PLAYER_NAMES)}

    # P x E matrix of projections
    M = df[event_cols].to_numpy(dtype=float)  # keeps NaN

    # ADP array
    ADP_ARR = df["ADP"].to_numpy(dtype=float) if "ADP" in df.columns else np.full(len(df), 999.0)

    # Event multipliers aligned to event_cols
    EVENT_MULT_ARR = np.array([ROUND_MULTIPLIERS[EVENT_TO_ROUND[e]] for e in event_cols], dtype=float)

    # Event weights aligned to event_cols
    EVENT_W_ARR = np.array([float(EVENT_WEIGHTS.get(e, 0.0)) for e in event_cols], dtype=float)

    return df

def set_contest_format(contest_name: str) -> None:
    """
    Set the contest format for the model.
    Must be called after init_model() to update event mappings and rebuild caches.
    """
    global EVENT_TO_ROUND, event_cols, EVENT_MULT_ARR, EVENT_W_ARR, M

    if contest_name not in CONTEST_FORMATS:
        raise ValueError(f"Unknown contest: {contest_name}. Available: {list(CONTEST_FORMATS.keys())}")

    EVENT_TO_ROUND = CONTEST_FORMATS[contest_name]

    # Update event_cols to only include events in this contest
    if df is not None:
        event_cols = [c for c in EVENT_TO_ROUND.keys() if c in df.columns]

        # Clean the event columns for this contest (they might have '-' strings)
        df[event_cols] = (
            df[event_cols]
            .replace('-', np.nan)
            .apply(pd.to_numeric, errors='coerce')
        )

        # Rebuild the M matrix with only the relevant events (now properly cleaned)
        M = df[event_cols].to_numpy(dtype=float)  # keeps NaN

        # Rebuild EVENT_MULT_ARR with new event list
        EVENT_MULT_ARR = np.array(
            [ROUND_MULTIPLIERS[EVENT_TO_ROUND[e]] for e in event_cols],
            dtype=float
        )

        # Rebuild EVENT_W_ARR with new event weights
        EVENT_W_ARR = np.array(
            [float(EVENT_WEIGHTS.get(e, 0.0)) for e in event_cols],
            dtype=float
        )

def reload_model_with_csv(csv_path: str, contest_name: str) -> None:
    """
    Reload the entire model with a new CSV file and contest format.
    This resets all cached data and rebuilds everything from scratch.
    """
    # Re-initialize with new CSV
    init_model(csv_path)

    # Set the contest format
    set_contest_format(contest_name)

def compute_event_weights(df, event_cols, alpha=0.5):
    avg_pts = df[event_cols].mean(skipna=True)
    U = (avg_pts - avg_pts.min()) / (avg_pts.max() - avg_pts.min())

    availability = df[event_cols].notna().sum()
    rarity_raw = availability.max() - availability
    R = (rarity_raw - rarity_raw.min()) / (rarity_raw.max() - rarity_raw.min())

    return (alpha * U + (1 - alpha) * R).fillna(0)

def best_6_event_score(players, event):
    scores = df.loc[players, event].dropna().values
    return np.sum(np.sort(scores)[-MIN_PLAYERS_PER_EVENT:]) if len(scores) else 0

def total_best_ball_score(players):
    total = 0

    for event in event_cols:
        event_points = best_6_event_score(players, event)

        # APPLY ROUND MULTIPLIER HERE
        round_id = EVENT_TO_ROUND[event]
        multiplier = ROUND_MULTIPLIERS[round_id]

        total += multiplier * event_points

    return total

def coverage_deficit(players, event):
    count = df.loc[players, event].notna().sum()
    return max(0, MIN_PLAYERS_PER_EVENT - count)

def classify_event_urgency(drafted_players):
    remaining_picks = TEAM_SIZE - len(drafted_players)
    urgency = {}

    for event in event_cols:
        covered = df.loc[drafted_players, event].notna().sum()
        deficit = MIN_PLAYERS_PER_EVENT - covered

        if deficit >= remaining_picks:
            urgency[event] = "urgent"
        elif deficit == remaining_picks - 1:
            urgency[event] = "soon"
        else:
            urgency[event] = "safe"

    return urgency

def player_event_impact(drafted_players, candidate, urgency):
    impacts = []

    urgent_icon = "⚠️"  # "‼", "◆", "★", "✖", "●", "⛔", "⚠️"
    # soon_icon = "○"

    for event, level in urgency.items():
        if level == "safe":
            continue

        if pd.notna(df.loc[candidate, event]):
            if level == "urgent":
                impacts.append(f"{urgent_icon} {event}")
            elif level == "soon":
                impacts.append(f"{event}")

    return " | ".join(impacts)

def roster_score(players, round_number, event_weights, urgency):
    points = total_best_ball_score(players)
    lambda_r = LAMBDA_PER_ROUND * round_number

    penalty = sum(
        event_weights[e] * coverage_deficit(players, e)
        for e in event_cols
    )

    bonus = 0
    for event, level in urgency.items():
        if level != "safe" and pd.notna(df.loc[players[-1], event]):
            bonus += URGENCY_MULTIPLIER[level]

    return points - lambda_r * penalty + bonus

def recommend_players_fast(drafted_players, unavailable_players, round_number):
    """
    Vectorized recommender.
    Keeps the same intent as roster_score: points - lambda*penalty + bonus,
    but computed using marginal deltas (base terms drop out).
    """
    if df is None or event_cols is None or M is None:
        raise RuntimeError("Model not initialized. Call init_model(csv_path) first.")

    # -------------------------
    # Prep indices
    # -------------------------
    drafted_idx = np.array([NAME_TO_I[p] for p in drafted_players], dtype=int) if drafted_players else np.array([], dtype=int)
    unavailable_set = set(unavailable_players) | set(drafted_players)

    avail_names = [n for n in PLAYER_NAMES if n not in unavailable_set]
    cand_idx = np.array([NAME_TO_I[n] for n in avail_names], dtype=int)

    # Candidate matrix: N x E
    C = M[cand_idx, :]  # float, NaN ok
    plays = ~np.isnan(C)  # N x E bool

    # -------------------------
    # Current roster per-event state
    # -------------------------
    if drafted_idx.size == 0:
        drafted_M = np.empty((0, len(event_cols)), dtype=float)
    else:
        drafted_M = M[drafted_idx, :]  # D x E

    drafted_plays = ~np.isnan(drafted_M)                     # D x E
    count = drafted_plays.sum(axis=0).astype(int)            # E
    deficit = np.maximum(0, MIN_PLAYERS_PER_EVENT - count)   # E

    # Threshold = current 6th-best (min of current top-6) when count >= 6
    # If count < 6, threshold is unused.
    threshold = np.zeros(len(event_cols), dtype=float)

    if drafted_M.shape[0] > 0:
        # Fill NaNs as -inf so they never appear in top-6
        dm = np.where(np.isnan(drafted_M), -np.inf, drafted_M)  # D x E

        # For each event, get 6th-largest among drafted => kth element of partition
        # If D < 6, partition still works but kth index must exist; guard with count>=6.
        for e in range(len(event_cols)):
            if count[e] >= MIN_PLAYERS_PER_EVENT:
                col = dm[:, e]
                # Take top-6 values via partition; then the min of that set is the threshold
                top6 = np.partition(col, -MIN_PLAYERS_PER_EVENT)[-MIN_PLAYERS_PER_EVENT:]
                threshold[e] = float(np.min(top6))
            else:
                threshold[e] = 0.0

    # -------------------------
    # Δpoints for candidates (best-ball marginal gain)
    # -------------------------
    # If count < 6: delta = candidate score
    # Else: delta = max(0, score - threshold)
    # Missing (NaN) => delta 0
    delta_event = np.where(
        plays,
        np.where(count < MIN_PLAYERS_PER_EVENT, C, np.maximum(0.0, C - threshold)),
        0.0
    )
    delta_points = (delta_event * EVENT_MULT_ARR).sum(axis=1)  # N

    # -------------------------
    # Penalty term (same spirit as roster_score penalty)
    # penalty(roster) = sum(w_e * deficit_e(roster))
    #
    # Adding a candidate reduces deficit by 1 for events where deficit>0 and candidate plays.
    # So Δpenalty = - sum(w_e * 1[deficit>0 and plays])
    # Score uses: -lambda * penalty, so improvement is +lambda * sum(...)
    # -------------------------
    lambda_r = LAMBDA_PER_ROUND * round_number

    helps_deficit = plays & (deficit > 0)  # N x E bool
    penalty_improve = (helps_deficit * EVENT_W_ARR).sum(axis=1)  # N
    penalty_term = lambda_r * penalty_improve  # add to score

    # -------------------------
    # Urgency bonus (same as your current bonus: +2 urgent, +1 soon per event played)
    # -------------------------
    urgency = classify_event_urgency(drafted_players)
    urgent_mask = np.array([urgency[e] == "urgent" for e in event_cols], dtype=bool)
    soon_mask = np.array([urgency[e] == "soon" for e in event_cols], dtype=bool)

    bonus = (
        plays[:, urgent_mask].sum(axis=1) * URGENCY_MULTIPLIER["urgent"]
        + plays[:, soon_mask].sum(axis=1) * URGENCY_MULTIPLIER["soon"]
    )

    # -------------------------
    # Final score for ranking (marginal form)
    # -------------------------
    score = delta_points + penalty_term + bonus

    # Build output rows (top N only)
    order = np.argsort(-score)[:TOP_N_RECOMMENDATIONS]

    # Event Impact strings are still easiest in Python (only for top N)
    rows = []
    for j in order:
        name = avail_names[j]
        rows.append({
            "Player": name,
            "Score": float(score[j]),
            "ADP": float(ADP_ARR[cand_idx[j]]),
            "Event Impact": player_event_impact(drafted_players, name, urgency)
        })

    return pd.DataFrame(rows).reset_index(drop=True)

# New function for The Albatross with plateau detection
def recommend_players_fast_albatross(drafted_players, unavailable_players, round_number):
    """
    Vectorized recommender for The Albatross.
    Uses plateau detection and returns individual event projections.
    """
    if df is None or event_cols is None or M is None:
        raise RuntimeError("Model not initialized. Call init_model(csv_path) first.")

    # -------------------------
    # Prep indices
    # -------------------------
    drafted_idx = np.array([NAME_TO_I[p] for p in drafted_players], dtype=int) if drafted_players else np.array([], dtype=int)
    unavailable_set = set(unavailable_players) | set(drafted_players)

    avail_names = [n for n in PLAYER_NAMES if n not in unavailable_set]
    cand_idx = np.array([NAME_TO_I[n] for n in avail_names], dtype=int)

    C = M[cand_idx, :]
    plays = ~np.isnan(C)

    # -------------------------
    # Current roster per-event state
    # -------------------------
    if drafted_idx.size == 0:
        drafted_M = np.empty((0, len(event_cols)), dtype=float)
    else:
        drafted_M = M[drafted_idx, :]

    drafted_plays = ~np.isnan(drafted_M)
    count = drafted_plays.sum(axis=0).astype(int)
    deficit = np.maximum(0, MIN_PLAYERS_PER_EVENT - count)

    threshold = np.zeros(len(event_cols), dtype=float)

    if drafted_M.shape[0] > 0:
        dm = np.where(np.isnan(drafted_M), -np.inf, drafted_M)

        for e in range(len(event_cols)):
            if count[e] >= MIN_PLAYERS_PER_EVENT:
                col = dm[:, e]
                top6 = np.partition(col, -MIN_PLAYERS_PER_EVENT)[-MIN_PLAYERS_PER_EVENT:]
                threshold[e] = float(np.min(top6))
            else:
                threshold[e] = 0.0

    # -------------------------
    # Δpoints for candidates
    # -------------------------
    delta_event = np.where(
        plays,
        np.where(count < MIN_PLAYERS_PER_EVENT, C, np.maximum(0.0, C - threshold)),
        0.0
    )
    delta_points = (delta_event * EVENT_MULT_ARR).sum(axis=1)

    # -------------------------
    # Fallback: projected total
    # -------------------------
    proj_total = np.nansum(C * EVENT_MULT_ARR, axis=1)

    # -------------------------
    # Penalty term
    # -------------------------
    lambda_r = LAMBDA_PER_ROUND * round_number

    helps_deficit = plays & (deficit > 0)
    penalty_improve = (helps_deficit * EVENT_W_ARR).sum(axis=1)
    penalty_term = lambda_r * penalty_improve

    # -------------------------
    # Urgency bonus
    # -------------------------
    urgency = classify_event_urgency(drafted_players)
    urgent_mask = np.array([urgency[e] == "urgent" for e in event_cols], dtype=bool)
    soon_mask = np.array([urgency[e] == "soon" for e in event_cols], dtype=bool)

    bonus = (
        plays[:, urgent_mask].sum(axis=1) * URGENCY_MULTIPLIER["urgent"]
        + plays[:, soon_mask].sum(axis=1) * URGENCY_MULTIPLIER["soon"]
    )

    # -------------------------
    # Final score with plateau detection
    # -------------------------
    score_marginal = delta_points + penalty_term + bonus

    if np.nanmax(delta_points) < MARGINAL_SWITCH_EPS:
        score = proj_total + (FALLBACK_PENALTY_WEIGHT * penalty_term) + (FALLBACK_BONUS_WEIGHT * bonus)
    else:
        score = score_marginal

    # -------------------------
    # Build output with individual event columns
    # -------------------------
    order = np.argsort(-score)[:TOP_N_RECOMMENDATIONS]

    rows = []
    for j in order:
        name = avail_names[j]
        cand_event_vals = C[j, :]

        row = {
            "Player": name,
            "Score": float(score[j]),
            "ADP": float(ADP_ARR[cand_idx[j]]),
        }

        # Add one column per event
        for ei, event in enumerate(event_cols):
            v = cand_event_vals[ei]
            row[event] = (None if np.isnan(v) else float(v))

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)

def compute_event_coverage(drafted_players):
    remaining = TEAM_SIZE - len(drafted_players)
    return pd.DataFrame([
        {
            "Event": e,
            "Covered": df.loc[drafted_players, e].notna().sum(),
            "Needed": MIN_PLAYERS_PER_EVENT,
            "Remaining Picks": remaining
        }
        for e in event_cols
    ]).sort_values(["Covered", "Event"])

def set_round_multipliers(r1: float, r2: float, r3: float, r4: float) -> None:
    """
    Update round multipliers and refresh EVENT_MULT_ARR.
    Cheap: O(#events). Does NOT invalidate heavy caches (M, NAME_TO_I, etc).
    """
    global EVENT_MULT_ARR

    # Refresh per-event multiplier array used by recommend_players_fast
    if event_cols is None:
        raise RuntimeError("Model not initialized. Call init_model(csv_path) first.")

    ROUND_MULTIPLIERS["Round1"] = float(r1)
    ROUND_MULTIPLIERS["Round2"] = float(r2)
    ROUND_MULTIPLIERS["Round3"] = float(r3)
    ROUND_MULTIPLIERS["Round4"] = float(r4)

    # Same logic as init_model()
    EVENT_MULT_ARR = np.array(
        [ROUND_MULTIPLIERS[EVENT_TO_ROUND[e]] for e in event_cols],
        dtype=float
    )
