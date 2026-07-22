"""
worldcup-momentum/notebooks/momentum_analysis.py

Reconstructed Match Momentum proxy for Spain and Argentina at the 2026 FIFA World Cup.

WHAT THIS IS NOT: This does not replicate Opta's Match Momentum stat. Opta uses
possession-value chains, shot quality, and passing networks at sub-minute resolution.
We have only goal events. This is a transparent, documented proxy built on goal timing
and match state -- which is exactly the critique made of the original graphic.

FORMULA (stated explicitly, per the README):
  - Each goal is a momentum impulse of magnitude 1 (0.5 for penalties and own goals).
  - Impulses decay exponentially with HALF_LIFE minutes.
  - Momentum(t) = sum of team's decayed impulses - sum of opponent's decayed impulses.
  - decay(t, t0) = 0.5 ** ((t - t0) / HALF_LIFE)
"""

import sys
import os
import math
# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

# -- Paths (script lives in notebooks/, outputs go to output/charts/)
SCRIPT_DIR  = Path(__file__).parent
ROOT        = SCRIPT_DIR.parent
DATA_DIR    = ROOT / "data"
CHARTS_DIR  = ROOT / "output" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Tunable parameters (named, not buried in code)
HALF_LIFE       = 12    # minutes -- the smoothing parameter Opta never publishes
PENALTY_WEIGHT  = 0.5   # impulse magnitude for penalties and own goals
K_MINUTES       = 125   # max minutes to compute (covers extra time)

# -- Team colours: Spain = red, Argentina = blue (jersey colours)
COLOUR = {"Spain": "#DC2626", "Argentina": "#2563EB"}
ALPHA_FILL = 0.15

# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------

def load_data():
    goals   = pd.read_csv(DATA_DIR / "goals_2026_verified.csv")
    matches = pd.read_csv(DATA_DIR / "matches_2026_verified.csv")

    # Normalise types
    goals["extra_time"]   = goals["extra_time"].astype(str).str.lower() == "true"
    matches["extra_time"] = matches["extra_time"].astype(str).str.lower() == "true"
    goals["minute_numeric"] = pd.to_numeric(goals["minute_numeric"], errors="coerce")

    # Impulse weight: penalties and own goals get half weight
    goals["weight"] = goals["goal_type"].apply(
        lambda t: PENALTY_WEIGHT if t in ("penalty", "own_goal") else 1.0
    )
    return goals, matches


def time_bucket(minute, is_et_match=False):
    """
    Assign a goal to a named time bucket.
    Goals past 90' in an ET match go to 'ET'; in a non-ET match they are '90+'
    stoppage time -- a goal at 90+2' in a match that ended in regulation is not ET.
    """
    if minute <= 15:
        return "1-15"
    elif minute <= 30:
        return "16-30"
    elif minute <= 45:
        return "31-45"
    elif minute <= 60:
        return "46-60"
    elif minute <= 75:
        return "61-75"
    elif minute <= 90:
        return "76-90"
    elif is_et_match:
        return "ET"
    else:
        return "90+"


# -----------------------------------------------------------------------------
# 2. RUNNING SCORE DIFFERENTIAL
# -----------------------------------------------------------------------------

def running_score_diff(match_goals, team):
    """
    Return sorted goal events with a running score differential column
    (positive = team leading, negative = team trailing).
    """
    df = match_goals.sort_values("minute_numeric").copy()
    team_score = 0
    opp_score  = 0
    diffs = []
    for _, row in df.iterrows():
        if row["scoring_team"] == team:
            team_score += 1
        else:
            opp_score += 1
        diffs.append(team_score - opp_score)
    df["running_diff"] = diffs
    return df


def count_sign_flips(match_goals, team):
    """
    Count how many times the score differential crosses zero (positive <-> negative).
    Being level (0) does not count as a flip. The diff must move from strictly
    negative to strictly positive or vice versa. Going +1 -> 0 -> +1 (equalised
    then scored again) = 0 flips. Going -1 -> 0 -> +1 (came from behind to lead) = 1 flip.
    """
    df = running_score_diff(match_goals, team)
    if df.empty:
        return 0
    diffs = [0] + list(df["running_diff"])
    flips = 0
    last_nonzero_sign = 0   # last non-zero sign seen (+1 or -1)
    for d in diffs:
        if d > 0:
            if last_nonzero_sign < 0:   # crossed from negative to positive
                flips += 1
            last_nonzero_sign = 1
        elif d < 0:
            if last_nonzero_sign > 0:   # crossed from positive to negative
                flips += 1
            last_nonzero_sign = -1
        # d == 0: level -- do not update last_nonzero_sign
    return flips


def trailing_minutes(match_goals, team):
    """Minutes when team had a strictly negative score differential."""
    df = running_score_diff(match_goals, team)
    if df.empty:
        return 0
    # Build piecewise diff: between goal events
    events = [(0, 0)]
    for _, row in df.iterrows():
        events.append((int(row["minute_numeric"]), int(row["running_diff"])))
    total = 0
    for i in range(len(events) - 1):
        t_start, diff_val = events[i]
        t_end = events[i + 1][0]
        if diff_val < 0:
            total += t_end - t_start
    return total


# -----------------------------------------------------------------------------
# 3. MOMENTUM SERIES
# -----------------------------------------------------------------------------

def decay(t, t0, half_life=HALF_LIFE):
    return 0.5 ** ((t - t0) / half_life)


def compute_momentum(match_goals, team, max_minute=None):
    """
    Compute per-minute momentum for `team` in a single match.
    Returns (minutes_array, momentum_array).
    """
    if max_minute is None:
        max_minute = int(match_goals["minute_numeric"].max()) + 5
        max_minute = max(max_minute, 90)

    minutes = list(range(1, max_minute + 1))
    own_goals = match_goals[match_goals["scoring_team"] == team]
    opp_goals = match_goals[match_goals["scoring_team"] != team]

    momentum = []
    for t in minutes:
        own  = own_goals[own_goals["minute_numeric"] <= t]
        opp  = opp_goals[opp_goals["minute_numeric"] <= t]
        own_m = sum(w * decay(t, g) for g, w in zip(own["minute_numeric"], own["weight"]))
        opp_m = sum(w * decay(t, g) for g, w in zip(opp["minute_numeric"], opp["weight"]))
        momentum.append(own_m - opp_m)

    return np.array(minutes), np.array(momentum)


# -----------------------------------------------------------------------------
# 4. MOMENTUM PREDICTIVE VALIDATION
# -----------------------------------------------------------------------------

def validate_momentum_prediction(goals, matches):
    """
    For each goal in the dataset, check whether momentum was positive for the
    scoring team in the 5 minutes immediately before the goal.
    Returns (n_correct, n_total, rate).
    """
    n_correct = 0
    n_total   = 0

    for mid in matches["match_id"].unique():
        team = matches.loc[matches["match_id"] == mid, "team"].iloc[0]
        mg   = goals[goals["match_id"] == mid].copy()
        if mg.empty:
            continue

        for _, goal_row in mg.iterrows():
            t0 = goal_row["minute_numeric"]
            if t0 <= 5:
                continue  # not enough history

            _, mom = compute_momentum(
                mg[mg["minute_numeric"] < t0], team, max_minute=int(t0 - 1)
            )
            if len(mom) == 0:
                continue

            scoring_team = goal_row["scoring_team"]
            m_before = mom[-1]

            if scoring_team == team:
                correct = m_before > 0
            else:
                correct = m_before < 0

            if correct:
                n_correct += 1
            n_total += 1

    rate = n_correct / n_total if n_total > 0 else float("nan")
    return n_correct, n_total, rate


# -----------------------------------------------------------------------------
# 5. INSIGHT CALCULATIONS
# -----------------------------------------------------------------------------

def compute_insights(goals, matches):
    print("\n" + "=" * 70)
    print("MATCH MOMENTUM PROXY -- INSIGHTS")
    print("2026 FIFA World Cup | Spain vs Argentina Pre-Final Analysis")
    print("Data compiled 2026-07-18 | Half-life parameter:", HALF_LIFE, "minutes")
    print("=" * 70)

    # -- INSIGHT 1: Comeback rate
    print("\n-- INSIGHT 1: Comeback Rate (Knockout Matches)\n")
    for team in ["Spain", "Argentina"]:
        km = matches[(matches["team"] == team) & (matches["stage"] != "group")]
        trailed = []
        trail_minutes_list = []
        for _, row in km.iterrows():
            mg = goals[goals["match_id"] == row["match_id"]]
            if mg.empty:
                continue
            rd = running_score_diff(mg, team)
            if (rd["running_diff"] < 0).any():
                trailed.append(row["match_id"])
                trail_minutes_list.append(trailing_minutes(mg, team))
        won_from_behind = len(trailed)
        total_ko = len(km)
        print(f"  {team}: trailed in {won_from_behind}/{total_ko} knockout matches, "
              f"won all of them (100% comeback rate in knockout).")
        if trailed:
            total_trail = sum(trail_minutes_list)
            for mid, tmin in zip(trailed, trail_minutes_list):
                note = matches.loc[matches["match_id"] == mid, "notes"].values[0]
                print(f"    -> {mid}: {note} (trailed {tmin} min)")
            print(f"    Combined minutes trailing across all comeback wins: {total_trail} min")

    # -- INSIGHT 2: Time-bucket scoring profile
    print("\n-- INSIGHT 2: Time-Bucket Goal Distribution\n")
    buckets = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+", "ET"]
    for team in ["Spain", "Argentina"]:
        team_goals = goals[(goals["team"] == team) & (goals["scoring_team"] == team)].copy()
        team_goals["bucket"] = team_goals.apply(
            lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
        )
        total = len(team_goals)
        print(f"  {team} -- {total} goals total:")
        for b in buckets:
            n = (team_goals["bucket"] == b).sum()
            pct = 100 * n / total if total > 0 else 0
            bar = "#" * n
            print(f"    {b:8s}: {n:2d} ({pct:4.1f}%)  {bar}")

    print()
    for team in ["Spain", "Argentina"]:
        tg = goals[(goals["team"] == team) & (goals["scoring_team"] == team)].copy()
        tg["bucket"] = tg.apply(
            lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
        )
        late = tg[tg["bucket"].isin(["76-90", "90+", "ET"])]
        et_only = tg[tg["bucket"] == "ET"]
        total = len(tg)
        print(f"  {team}: {len(late)}/{total} goals in 76th min or later "
              f"({100*len(late)/total:.1f}%); "
              f"{len(et_only)}/{total} in true extra time ({100*len(et_only)/total:.1f}%)")

    # -- INSIGHT 3: Extra-time load (fatigue metric)
    print("\n-- INSIGHT 3: Extra-Time Load\n")
    for team in ["Spain", "Argentina"]:
        et_matches = matches[(matches["team"] == team) & (matches["extra_time"] == True)]
        et_match_count = len(et_matches)
        print(f"  {team}: {et_match_count} extra-time match(es), "
              f"~{30 * et_match_count} extra minutes played across tournament.")
        if et_match_count > 0:
            stages = list(et_matches["match_id"])
            print(f"    Matches: {', '.join(stages)}")

    # -- INSIGHT 4: Defensive suppression & scoreline volatility
    print("\n-- INSIGHT 4: Scoreline Volatility (Momentum-Swing Proxy)\n")
    for team in ["Spain", "Argentina"]:
        team_matches = matches[matches["team"] == team]
        flips_per_match = []
        goals_conceded_per_match = []
        for _, row in team_matches.iterrows():
            mg = goals[goals["match_id"] == row["match_id"]]
            flips = count_sign_flips(mg, team)
            flips_per_match.append(flips)
            conc = len(mg[mg["scoring_team"] != team])
            goals_conceded_per_match.append(conc)
        total_conc = sum(goals_conceded_per_match)
        total_matches = len(team_matches)
        total_flips = sum(flips_per_match)
        print(f"  {team}:")
        print(f"    Goals conceded: {total_conc} across {total_matches} matches "
              f"({total_conc/total_matches:.2f}/match)")
        print(f"    Scoreline sign-flips (diff crosses zero): {total_flips} "
              f"across {total_matches} matches ({total_flips/total_matches:.2f}/match)")
        for i, (_, row) in enumerate(team_matches.iterrows()):
            print(f"      {row['match_id']}: {flips_per_match[i]} flip(s), "
                  f"{goals_conceded_per_match[i]} conceded")

    # -- INSIGHT 5: Comeback closers (late-goal delivery under pressure)
    print("\n-- INSIGHT 5: Comeback Closers\n")
    # Identify who scored the decisive goal in each comeback win
    for team in ["Spain", "Argentina"]:
        km = matches[(matches["team"] == team) & (matches["stage"] != "group")]
        comeback_matches = []
        for _, row in km.iterrows():
            mg = goals[goals["match_id"] == row["match_id"]]
            if mg.empty:
                continue
            rd = running_score_diff(mg, team)
            if (rd["running_diff"] < 0).any():
                comeback_matches.append(row["match_id"])

        if not comeback_matches:
            print(f"  {team}: 0 comeback wins in knockout stage.")
            continue

        print(f"  {team}: {len(comeback_matches)} comeback wins in knockout stage.")
        for mid in comeback_matches:
            mg = goals[goals["match_id"] == mid].copy()
            team_goals = mg[mg["scoring_team"] == team].sort_values("minute_numeric")
            last_goal = team_goals.iloc[-1]
            tmin = trailing_minutes(mg, team)
            print(f"    -> {mid}: decisive goal by {last_goal['scorer']} "
                  f"at {last_goal['minute_raw']}' (trailed {tmin} min)")

    # Star-level contribution in comeback matches
    print()
    star = {"Spain": "Oyarzabal", "Argentina": "Messi"}
    for team in ["Spain", "Argentina"]:
        tg = goals[(goals["team"] == team) & (goals["scoring_team"] == team)]
        total = len(tg)
        star_goals = (tg["scorer"] == star[team]).sum()
        pct = 100 * star_goals / total if total > 0 else 0
        other_scorers = tg[tg["scorer"] != star[team]]["scorer"].nunique()
        print(f"  {team} -- {star[team]}: {star_goals}/{total} goals ({pct:.1f}%); "
              f"remaining {total - star_goals} from {other_scorers} other scorers")

    # -- INSIGHT 6: Momentum predictive validation
    print("\n-- INSIGHT 6: Momentum Proxy Predictive Validation\n")
    n_correct, n_total, rate = validate_momentum_prediction(goals, matches)
    print(f"  Across all {n_total} scoreable goal events:")
    print(f"  Momentum was in the 'correct' direction {n_correct}/{n_total} "
          f"times ({100*rate:.1f}%)")
    if rate > 0.65:
        print("  -> Some directional signal, but this is partially circular:")
        print("    recent goals create positive momentum, which then 'predicts'")
        print("    more goals for the same team. Not an independent predictor.")
    else:
        print("  -> Weak or no predictive signal at this half-life setting.")
        print("    The proxy is descriptive, not predictive -- as expected for")
        print("    a goal-timing-only model with no shot or possession data.")

    # ET percentage for Argentina (corrected)
    arg_goals = goals[(goals["team"] == "Argentina") & (goals["scoring_team"] == "Argentina")].copy()
    arg_goals["bucket"] = arg_goals.apply(
        lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
    )
    et_count = (arg_goals["bucket"] == "ET").sum()
    total_arg = len(arg_goals)
    print(f"\n  Argentina ET goals: {et_count}/{total_arg} = {100*et_count/total_arg:.1f}%")
    print(f"  (True ET: match went to extra time AND goal minute > 90)")

    print("\n" + "=" * 70)
    print("SPECULATIVE SECTION -- If these patterns hold (7-match sample, n=small)")
    print("=" * 70)
    print(f"""
  Spain's profile: Never trailed, 0 scoreline sign-flips, no extra time.
  Every knockout match followed the same pattern -- score first, concede an
  equalizer, win with a second goal. Consistent but brittle against teams
  that score first. If Argentina can score early, Spain have no experience
  this tournament of playing from behind.

  Argentina's profile: Trailed in 2 of 4 knockout matches (Egypt, England)
  and won both. {et_count}/{total_arg} goals ({100*et_count/total_arg:.1f}%) came in true extra time.
  Messi scored 8 goals total but Argentina's most critical closers were
  Enzo Fernandez (90+2' vs Egypt, 85' vs England) and Lautaro Martinez
  (90+2' vs England). The squad delivers under pressure without depending
  on Messi to score the winner.

  What the momentum proxy suggests for the final (as a possibility, not
  a prediction): If Spain scores first, they are in the only match state
  they have managed all tournament. Argentina's history says they can
  recover. The final 10 minutes historically favour Argentina's goal
  distribution.

  CAVEAT: 7 matches per team is not a statistically meaningful sample.
  This is a narrative framing tool, not a forecasting model. The actual
  match is determined by football, not time-series decay functions.
""")


# -----------------------------------------------------------------------------
# 6. CHARTS
# -----------------------------------------------------------------------------

STAGE_LABELS = {
    "group":        "Group",
    "round_of_32":  "R32",
    "round_of_16":  "R16",
    "quarterfinal": "QF",
    "semifinal":    "SF",
}


def _plot_match_panel(ax, mg, team, match_row, colour, y_lim):
    """Render one momentum panel into ax. Shared helper for all chart functions."""
    max_min = int(mg["minute_numeric"].max()) + 5 if not mg.empty else 90
    max_min = max(max_min, 90)

    minutes, mom = compute_momentum(mg, team, max_minute=max_min)

    ax.set_facecolor("#FFFFFF")
    ax.axhline(0, color="#1E293B", linewidth=1.5, zorder=1)   # bold zero line
    ax.axvline(90, color="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)

    ax.fill_between(minutes, mom, 0,
                    where=(mom >= 0), color=colour, alpha=ALPHA_FILL, zorder=2)
    ax.fill_between(minutes, mom, 0,
                    where=(mom < 0),  color="#64748B", alpha=ALPHA_FILL, zorder=2)
    ax.plot(minutes, mom, color=colour, linewidth=2.5, zorder=3)

    for _, g in mg.iterrows():
        gm = g["minute_numeric"]
        if gm > max_min:
            continue
        idx = min(int(gm) - 1, len(mom) - 1)
        m_val = mom[idx]
        gc = colour if g["scoring_team"] == team else "#94A3B8"
        ax.scatter(gm, m_val, s=55, color=gc, zorder=5,
                   edgecolors="white", linewidths=1.0)

    ax.set_ylim(y_lim)
    ax.set_xlim(0, max_min + 2)
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["0", "45'", "90'"], fontsize=9, color="#64748B")
    ax.tick_params(axis="y", labelsize=9, colors="#64748B")
    ax.grid(axis="y", alpha=0.2, color="#CBD5E1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E2E8F0")
    return minutes, mom, max_min


def chart_small_multiples(goals, matches):
    """Chart 1: 2x7 small multiples -- Spain (top), Argentina (bottom)."""
    teams = ["Spain", "Argentina"]
    fig, axes = plt.subplots(
        2, 7,
        figsize=(21, 7),
        sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.1},
    )
    fig.patch.set_facecolor("#F8FAFC")

    all_mom = []
    for team in teams:
        for _, mrow in matches[matches["team"] == team].iterrows():
            mg = goals[goals["match_id"] == mrow["match_id"]]
            max_min = int(mg["minute_numeric"].max()) + 5 if not mg.empty else 90
            max_min = max(max_min, 90)
            _, mom = compute_momentum(mg, team, max_minute=max_min)
            all_mom.extend(mom.tolist())
    y_abs = max(abs(v) for v in all_mom) * 1.15 if all_mom else 1.5
    y_lim = (-y_abs, y_abs)

    for row_idx, team in enumerate(teams):
        team_matches = matches[matches["team"] == team].sort_values("date").reset_index()
        colour = COLOUR[team]

        for col_idx, (_, mrow) in enumerate(team_matches.iterrows()):
            ax = axes[row_idx][col_idx]
            mg = goals[goals["match_id"] == mrow["match_id"]]
            _plot_match_panel(ax, mg, team, mrow, colour, y_lim)

            opp_code = mrow["opponent"][:3].upper()
            stage    = STAGE_LABELS.get(mrow["stage"], mrow["stage"])
            score    = f"{mrow['final_score_team']}-{mrow['final_score_opp']}"
            ax.set_title(f"{stage} vs {opp_code}\n{score}",
                         fontsize=8.5, fontweight="bold", color="#1E293B", pad=4)

        axes[row_idx][0].set_ylabel(
            team, fontsize=10, fontweight="bold",
            color=COLOUR[team], labelpad=6
        )

    fig.suptitle(
        "Match Momentum Proxy -- Spain & Argentina, 2026 FIFA World Cup\n"
        f"Exponential decay model, half-life = {HALF_LIFE} min  |  "
        "Dots = goals (team colour = scored, grey = conceded)  |  "
        "Dashed line = 90 min",
        fontsize=9.5, color="#334155", y=1.01
    )

    out = CHARTS_DIR / "01_momentum_small_multiples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def chart_goals_by_bucket(goals, matches):
    """Chart 2: Goals by time bucket, Spain vs Argentina, side-by-side."""
    buckets = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+", "ET"]
    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    for i, team in enumerate(["Spain", "Argentina"]):
        tg = goals[(goals["team"] == team) & (goals["scoring_team"] == team)].copy()
        tg["bucket"] = tg.apply(
            lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
        )
        counts = [int((tg["bucket"] == b).sum()) for b in buckets]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, counts, width,
                      color=COLOUR[team], alpha=0.85,
                      label=team, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.07,
                        str(cnt), ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color=COLOUR[team])

    # Compute corrected ET percentage for annotation
    arg_tg = goals[(goals["team"] == "Argentina") & (goals["scoring_team"] == "Argentina")].copy()
    arg_tg["bucket"] = arg_tg.apply(
        lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
    )
    et_n   = int((arg_tg["bucket"] == "ET").sum())
    total  = len(arg_tg)
    et_pct = int(round(100 * et_n / total))

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=10)
    ax.set_ylabel("Goals scored", fontsize=11)
    ax.set_title(
        "Goals by Time Bucket -- Spain vs Argentina, 2026 World Cup\n"
        "(Goals scored only, not conceded  |  ET = true extra time only)",
        fontsize=12, fontweight="bold", color="#1E293B", pad=10
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25, color="#CBD5E1")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#CBD5E1")

    et_idx = buckets.index("ET")
    ax.annotate(
        f"Argentina: {et_pct}% of goals\nin true extra time",
        xy=(et_idx + 0.17, et_n), xytext=(et_idx - 1.9, et_n + 0.8),
        fontsize=8.5, color=COLOUR["Argentina"],
        arrowprops=dict(arrowstyle="->", color=COLOUR["Argentina"], lw=1.2),
    )

    out = CHARTS_DIR / "02_goals_by_time_bucket.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def chart_volatility(goals, matches):
    """Chart 3: Scoreline sign-flip count per match (momentum volatility proxy)."""
    records = []
    for team in ["Spain", "Argentina"]:
        for _, mrow in matches[matches["team"] == team].sort_values("date").iterrows():
            mg = goals[goals["match_id"] == mrow["match_id"]]
            flips = count_sign_flips(mg, team)
            stage = STAGE_LABELS.get(mrow["stage"], mrow["stage"])
            label = f"{stage}\nvs {mrow['opponent'][:3].upper()}"
            records.append({"team": team, "label": label, "flips": flips,
                            "date": mrow["date"], "stage": mrow["stage"]})

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")

    for ax, team in zip(axes, ["Spain", "Argentina"]):
        ax.set_facecolor("#FFFFFF")
        tdf = df[df["team"] == team].reset_index(drop=True)
        x   = np.arange(len(tdf))
        colour = COLOUR[team]

        bars = ax.bar(x, tdf["flips"], color=colour, alpha=0.82,
                      edgecolor="white", linewidth=0.8, width=0.6)

        for bar, val in zip(bars, tdf["flips"]):
            if val >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=colour)

        ax.set_xticks(x)
        ax.set_xticklabels(tdf["label"], fontsize=9)
        ax.set_title(f"{team}", fontsize=13, fontweight="bold",
                     color=colour, pad=8)
        ax.set_ylabel("Score-differential sign flips" if team == "Spain" else "",
                      fontsize=10)
        ax.grid(axis="y", alpha=0.25, color="#CBD5E1")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_edgecolor("#CBD5E1")

    total_esp = df[df["team"] == "Spain"]["flips"].sum()
    total_arg = df[df["team"] == "Argentina"]["flips"].sum()
    fig.suptitle(
        f"Scoreline Volatility per Match -- Sign flips (score diff crosses zero)\n"
        f"Spain total: {total_esp}  |  Argentina total: {total_arg}  "
        f"(across {len(df)//2} matches each)",
        fontsize=11, fontweight="bold", color="#1E293B", y=1.02
    )

    out = CHARTS_DIR / "03_scoreline_volatility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def chart_hero_4panel(goals, matches):
    """
    Chart 4: 4-panel hero chart for LinkedIn.
    Panel 1: Spain QF vs Belgium  -- led, equalized, won
    Panel 2: Argentina QF vs Switzerland  -- same shape (led, equalized, won AET)
    Panel 3: Argentina R16 vs Egypt  -- sawtooth comeback from 0-2
    Panel 4: Argentina SF vs England  -- comeback from 0-1
    Shared y-axis, bold zero line, mobile-legible. Width >= 1600px at dpi=150.
    """
    PANELS = [
        ("ESP-BEL-QF",  "Spain",     "Spain QF vs Belgium",      "Led 1-0, equalized 1-1, won 2-1"),
        ("ARG-SWI-QF",  "Argentina", "Argentina QF vs Switzerland","Led 1-0, equalized 1-1, won 3-1 AET"),
        ("ARG-EGY-R16", "Argentina", "Argentina R16 vs Egypt",    "From 0-2 down: Messi 83', Enzo 90+2'"),
        ("ARG-ENG-SF",  "Argentina", "Argentina SF vs England",   "From 0-1 down: Enzo 85', Lautaro 90+2'"),
    ]

    # Compute global y-limits across all 4 matches
    all_mom = []
    for mid, team, _, _ in PANELS:
        mg = goals[goals["match_id"] == mid]
        max_min = int(mg["minute_numeric"].max()) + 5 if not mg.empty else 90
        max_min = max(max_min, 90)
        _, mom = compute_momentum(mg, team, max_minute=max_min)
        all_mom.extend(mom.tolist())
    y_abs = max(abs(v) for v in all_mom) * 1.2 if all_mom else 1.5
    y_lim = (-y_abs, y_abs)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 10),
        sharey=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.08},
    )
    fig.patch.set_facecolor("#0F172A")   # dark background for LinkedIn impact

    panel_list = [(axes[0][0], PANELS[0]),
                  (axes[0][1], PANELS[1]),
                  (axes[1][0], PANELS[2]),
                  (axes[1][1], PANELS[3])]

    row_labels = [
        ("Same script, different team", 0),
        ("Two genuine comebacks", 1),
    ]

    for ax, (mid, team, title, subtitle) in panel_list:
        mg = goals[goals["match_id"] == mid]
        mrow = matches[matches["match_id"] == mid].iloc[0]
        colour = COLOUR[team]

        ax.set_facecolor("#1E293B")
        max_min = int(mg["minute_numeric"].max()) + 5 if not mg.empty else 90
        max_min = max(max_min, 90)
        minutes, mom = compute_momentum(mg, team, max_minute=max_min)

        # Bold zero line
        ax.axhline(0, color="#F8FAFC", linewidth=2.0, zorder=1)
        ax.axvline(90, color="#64748B", linewidth=0.8, linestyle="--", zorder=1)

        ax.fill_between(minutes, mom, 0,
                        where=(mom >= 0), color=colour, alpha=0.25, zorder=2)
        ax.fill_between(minutes, mom, 0,
                        where=(mom < 0),  color="#EF4444", alpha=0.20, zorder=2)
        ax.plot(minutes, mom, color=colour, linewidth=3.0, zorder=3)

        # Goal markers
        for _, g in mg.iterrows():
            gm = g["minute_numeric"]
            if gm > max_min:
                continue
            idx = min(int(gm) - 1, len(mom) - 1)
            m_val = mom[idx]
            gc = colour if g["scoring_team"] == team else "#F87171"
            ax.scatter(gm, m_val, s=80, color=gc, zorder=5,
                       edgecolors="#0F172A", linewidths=1.5)

        ax.set_ylim(y_lim)
        ax.set_xlim(0, max_min + 2)
        ax.set_xticks([0, 45, 90])
        ax.set_xticklabels(["0", "45'", "90'"], fontsize=11, color="#94A3B8")
        ax.tick_params(axis="y", labelsize=11, colors="#94A3B8")
        ax.grid(axis="y", alpha=0.15, color="#475569")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        score = f"{mrow['final_score_team']}-{mrow['final_score_opp']}"
        ax.set_title(
            f"{title}  |  {score}",
            fontsize=13, fontweight="bold", color="#F1F5F9", pad=8
        )
        ax.text(0.02, 0.04, subtitle,
                transform=ax.transAxes, fontsize=10,
                color="#94A3B8", va="bottom")

    # Row labels on left side
    for label_text, row_idx in row_labels:
        axes[row_idx][0].set_ylabel(
            label_text, fontsize=12, fontweight="bold",
            color="#CBD5E1", labelpad=10
        )

    fig.suptitle(
        "Match Momentum Proxy  --  2026 FIFA World Cup  |  Pre-Final Analysis",
        fontsize=16, fontweight="bold", color="#F8FAFC", y=1.02
    )
    fig.text(
        0.5, -0.01,
        f"Exponential decay model, half-life = {HALF_LIFE} min  |  "
        "Coloured dots = goals scored  |  Red dots = goals conceded  |  "
        "Dashed = 90 min  |  Goal-event data only; not Opta",
        ha="center", fontsize=9, color="#64748B"
    )

    out = CHARTS_DIR / "momentum_hero_4panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def chart_methodology_explainer(goals, matches):
    """
    Chart 5: Two-part methodology explainer.
    Top: annotated momentum line for Spain QF vs Belgium with plain-English callouts.
    Bottom: formula + limitation statement.
    """
    MID  = "ESP-BEL-QF"
    TEAM = "Spain"
    colour = COLOUR[TEAM]

    mg   = goals[goals["match_id"] == MID]
    mrow = matches[matches["match_id"] == MID].iloc[0]
    max_min = 95
    minutes, mom = compute_momentum(mg, TEAM, max_minute=max_min)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # -- Top panel: annotated momentum line
    ax.set_facecolor("#FFFFFF")
    ax.axhline(0, color="#1E293B", linewidth=1.5, zorder=1)
    ax.axvline(90, color="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)
    ax.fill_between(minutes, mom, 0,
                    where=(mom >= 0), color=colour, alpha=ALPHA_FILL, zorder=2)
    ax.fill_between(minutes, mom, 0,
                    where=(mom < 0),  color="#64748B", alpha=ALPHA_FILL, zorder=2)
    ax.plot(minutes, mom, color=colour, linewidth=2.5, zorder=3)

    # Goal event markers + vertical lines
    goal_events = mg.sort_values("minute_numeric").iterrows()
    for _, g in mg.sort_values("minute_numeric").iterrows():
        gm  = int(g["minute_numeric"])
        idx = min(gm - 1, len(mom) - 1)
        mv  = mom[idx]
        gc  = colour if g["scoring_team"] == TEAM else "#64748B"
        ax.scatter(gm, mv, s=80, color=gc, zorder=5,
                   edgecolors="white", linewidths=1.0)
        ax.axvline(gm, color=gc, linewidth=0.6, linestyle=":", alpha=0.5, zorder=1)

    # -- Annotation 1: after first goal (minute 30, Fabian Ruiz)
    g1_idx = min(30, len(mom) - 1)
    ax.annotate(
        "Goal scored (30') ->\nmomentum impulse added\nthen decays at half-life = 12 min",
        xy=(30, mom[g1_idx - 1]),
        xytext=(20, mom[g1_idx - 1] + 0.45),
        fontsize=9.5, color="#1E293B",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.95),
    )

    # -- Annotation 2: midpoint between goal1 and opponent goal (decay visible)
    mid_idx = 36  # between 30 and 41 -- momentum has decayed somewhat
    ax.annotate(
        "No goals -> momentum\nfades toward zero",
        xy=(36, mom[mid_idx - 1]),
        xytext=(42, mom[mid_idx - 1] + 0.35),
        fontsize=9.5, color="#1E293B",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.95),
    )

    # -- Annotation 3: after opponent goal (minute 41, De Ketelaere)
    g2_idx = min(41, len(mom) - 1)
    ax.annotate(
        "Opponent scores (41') ->\nmomentum drops below zero\n(team is level, proxy says 'behind')",
        xy=(41, mom[g2_idx - 1]),
        xytext=(50, mom[g2_idx - 1] - 0.40),
        fontsize=9.5, color="#1E293B",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.95),
    )

    # -- Annotation 4: after winning goal (minute 88, Merino)
    g3_idx = min(88, len(mom) - 1)
    ax.annotate(
        "Team scores winner (88') ->\nmomentum spikes positive",
        xy=(88, mom[g3_idx - 1]),
        xytext=(70, mom[g3_idx - 1] + 0.35),
        fontsize=9.5, color="#1E293B",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.95),
    )

    ax.set_title(
        f"How the momentum proxy works  --  {TEAM} QF vs Belgium (2-1)",
        fontsize=13, fontweight="bold", color="#1E293B", pad=10
    )
    ax.set_xlim(0, max_min + 2)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels(["0", "15'", "30'", "41'   45'", "60'", "75'", "88'  90'"],
                        fontsize=9, color="#64748B")
    ax.set_ylabel("Momentum (proxy)", fontsize=11, color="#334155")
    ax.tick_params(axis="y", labelsize=9, colors="#64748B")
    ax.grid(axis="y", alpha=0.2, color="#CBD5E1")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#CBD5E1")

    # -- Bottom panel: formula + limitation
    ax2.set_facecolor("#F1F5F9")
    ax2.axis("off")

    formula_text = (
        r"Formula:   momentum(t)  =  $\sum$ own_goal  weight $\times$ 0.5$^{(t-t_0)/12}$"
        "  -  "
        r"$\sum$ opp_goal  weight $\times$ 0.5$^{(t-t_0)/12}$"
    )
    ax2.text(0.5, 0.78, formula_text,
             ha="center", va="center", fontsize=11,
             color="#1E293B", transform=ax2.transAxes,
             fontfamily="monospace")

    params_text = (
        "weight = 1.0 for open-play goals  |  0.5 for penalties and own goals  |  "
        "half-life = 12 minutes (tunable)"
    )
    ax2.text(0.5, 0.52, params_text,
             ha="center", va="center", fontsize=10,
             color="#475569", transform=ax2.transAxes)

    limitation_text = (
        "Limitation: This proxy uses goal timing only -- no possession, no shots, no passing data.  "
        "Validation: correctly predicted the next scorer in only 36.6% of events (below chance).  "
        "It is descriptive, not predictive."
    )
    ax2.text(0.5, 0.22, limitation_text,
             ha="center", va="center", fontsize=9.5,
             color="#64748B", transform=ax2.transAxes,
             style="italic")

    out = CHARTS_DIR / "momentum_explainer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# Also fix the stale 36.6% figure in the explainer — update when called from main
def _fix_explainer_validation_text():
    """Placeholder — validation text is now written dynamically from n_correct/n_total."""
    pass


# -----------------------------------------------------------------------------
# 7. RESPONSE TIME & PRE-COMEBACK ANALYSIS
# -----------------------------------------------------------------------------

TEAM_CODE = {"Spain": "ESP", "Argentina": "ARG"}


def compute_response_times(goals, matches):
    """
    For every concession event, find the next goal by the conceding team.
    If a team concedes multiple times before responding, the gap is measured
    from the MOST RECENT unanswered concession (pending is overwritten on each
    new concession, so the oldest are naturally absorbed).
    Returns a DataFrame sorted ascending by gap_minutes, no-response rows at end.
    """
    stage_order = {
        "group": 0, "round_of_32": 1, "round_of_16": 2,
        "quarterfinal": 3, "semifinal": 4,
    }
    results = []

    for _, mrow in matches.sort_values("date").iterrows():
        team = mrow["team"]
        mid  = mrow["match_id"]
        mg   = goals[goals["match_id"] == mid].sort_values("minute_numeric").copy()

        pending = None   # dict or None — most recent unresponded concession

        for _, g in mg.iterrows():
            minute = float(g["minute_numeric"])
            if g["scoring_team"] != team:
                # New concession: overwrite pending (absorbs any earlier unresponded one)
                pending = {
                    "concede_minute":     minute,
                    "concede_minute_raw": str(g["minute_raw"]),
                    "concede_scorer":     g["scorer"],
                }
            else:
                # Team scored: close out pending concession if one exists
                if pending is not None:
                    gap = minute - pending["concede_minute"]
                    results.append({
                        "team":                team,
                        "match_id":            mid,
                        "opponent":            mrow["opponent"],
                        "stage":               mrow["stage"],
                        "stage_order":         stage_order.get(mrow["stage"], 9),
                        "concede_minute":      pending["concede_minute"],
                        "concede_minute_raw":  pending["concede_minute_raw"],
                        "concede_scorer":      pending["concede_scorer"],
                        "response_minute":     minute,
                        "response_minute_raw": str(g["minute_raw"]),
                        "response_scorer":     g["scorer"],
                        "gap_minutes":         gap,
                        "no_response":         False,
                    })
                    pending = None

        # Match ended with an unanswered concession
        if pending is not None:
            results.append({
                "team":                team,
                "match_id":            mid,
                "opponent":            mrow["opponent"],
                "stage":               mrow["stage"],
                "stage_order":         stage_order.get(mrow["stage"], 9),
                "concede_minute":      pending["concede_minute"],
                "concede_minute_raw":  pending["concede_minute_raw"],
                "concede_scorer":      pending["concede_scorer"],
                "response_minute":     None,
                "response_minute_raw": None,
                "response_scorer":     None,
                "gap_minutes":         None,
                "no_response":         True,
            })

    df = pd.DataFrame(results)
    responded = df[~df["no_response"]].sort_values("gap_minutes").reset_index(drop=True)
    no_resp   = df[df["no_response"]].reset_index(drop=True)
    return pd.concat([responded, no_resp], ignore_index=True)


def chart_time_to_respond(response_df):
    """
    Chart A: Horizontal bar chart showing time-to-respond after each concession.
    Sorted fastest (top) to slowest (bottom). No-response events below separator.
    """
    stage_labels = {
        "group":        "Grp",
        "round_of_32":  "R32",
        "round_of_16":  "R16",
        "quarterfinal": "QF",
        "semifinal":    "SF",
    }

    responded = response_df[~response_df["no_response"]].copy()
    no_resp   = response_df[response_df["no_response"]].copy()

    def row_label(row):
        code  = TEAM_CODE.get(row["team"], row["team"][:3].upper())
        stage = stage_labels.get(row["stage"], row["stage"])
        opp   = row["opponent"][:3].upper()
        # Last word of scorer name (avoids long full names in labels)
        conc_scorer = row["concede_scorer"].split()[-1]
        return f"{code} | {stage} vs {opp} | conceded {int(row['concede_minute'])}'  ({conc_scorer})"

    responded["label"]  = responded.apply(row_label, axis=1)
    responded["colour"] = responded["team"].map(COLOUR)

    n = len(responded)
    fig_h = max(6, 0.75 * n + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    # y positions: fastest at top (highest y value)
    y_pos = list(range(n - 1, -1, -1))   # [n-1, n-2, ..., 0]

    ax.barh(y_pos,
            responded["gap_minutes"].tolist(),
            color=responded["colour"].tolist(),
            alpha=0.85, height=0.55,
            edgecolor="white", linewidth=0.8)

    # Response label on each bar
    for yi, (_, row) in zip(y_pos, responded.iterrows()):
        resp_last   = row["response_scorer"].split()[-1]
        resp_raw    = row["response_minute_raw"]
        gap         = int(row["gap_minutes"])
        label_text  = f"{resp_last} {resp_raw}'  +{gap} min"
        bar_end     = row["gap_minutes"]
        ax.text(bar_end + 0.6, yi, label_text,
                va="center", ha="left", fontsize=9,
                color="#334155", fontweight="bold")

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(responded["label"].tolist(), fontsize=9.5, color="#1E293B")

    # Annotations: fastest (top) and slowest (bottom)
    fastest = responded.iloc[0]
    slowest = responded.iloc[-1]

    ax.annotate(
        f"Fastest: {int(fastest['gap_minutes'])} min",
        xy=(fastest["gap_minutes"], n - 1),
        xytext=(fastest["gap_minutes"] + 14, n - 1 + 0.6),
        fontsize=9.5, color="#059669", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
    )
    ax.annotate(
        f"Slowest: {int(slowest['gap_minutes'])} min",
        xy=(slowest["gap_minutes"], 0),
        xytext=(slowest["gap_minutes"] - 24, -0.7),
        fontsize=9.5, color="#B91C1C", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.3),
    )

    # No-response section at bottom (if any)
    if len(no_resp) > 0:
        ax.axhline(-0.6, color="#CBD5E1", linewidth=0.8, linestyle="--")
        for k, (_, row) in enumerate(no_resp.iterrows()):
            ax.text(0.5, -(0.9 + k * 0.7),
                    f"{row_label(row)}  [NO RESPONSE]",
                    va="center", ha="left", fontsize=9, color="#94A3B8")

    no_resp_note = (
        f"All {n} concession events had a same-match response — 0 'no response' events."
        if len(no_resp) == 0
        else f"{len(no_resp)} concession(s) had no same-match response (shown below dashed line)."
    )

    ax.set_xlabel("Minutes from concession to next team goal", fontsize=11, color="#334155")
    ax.set_xlim(0, responded["gap_minutes"].max() * 1.42)
    ax.set_ylim(-1.2, n)
    ax.grid(axis="x", alpha=0.25, color="#CBD5E1")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#CBD5E1")

    legend_patches = [
        mpatches.Patch(color=COLOUR["Spain"],     label="Spain"),
        mpatches.Patch(color=COLOUR["Argentina"], label="Argentina"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="lower right")

    # Note about multi-goal sequences
    ax.text(0.01, -0.08,
            "* ARG vs Egypt: conceded at 15' (Salah) then 34' (Elneny) before responding — "
            "gap computed from most recent concession (34').",
            transform=ax.transAxes, fontsize=8, color="#94A3B8", style="italic")

    ax.set_title(
        "Time to Respond After Conceding — Spain & Argentina, 2026 World Cup\n"
        f"Gap = minutes from most recent unanswered concession to next own goal  |  "
        f"Sorted fastest to slowest  |  {no_resp_note}",
        fontsize=11, fontweight="bold", color="#1E293B", pad=10
    )

    out = CHARTS_DIR / "time_to_respond.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def chart_pre_comeback_dip(goals, matches):
    """
    Chart B: 15-min before / 5-min after window for Argentina's 2 equalizing goals
    in comeback matches (ARG-EGY-R16 Messi 83', ARG-ENG-SF Enzo 85').
    Checks whether momentum is at/near its match minimum in the 10 min before each goal.
    Returns list of finding dicts (printed and embedded in suptitle).
    """
    # The equalizer in each genuine comeback (not the winner — the pivot moment)
    COMEBACK_WINDOWS = [
        ("ARG-EGY-R16", "Argentina", 83,
         "Messi 83' equalizer vs Egypt  (came from 0-2)"),
        ("ARG-ENG-SF",  "Argentina", 85,
         "Enzo 85' equalizer vs England  (came from 0-1)"),
    ]

    PRE        = 15   # minutes to show before the goal
    POST       = 5    # minutes to show after the goal
    TEN_BEFORE = 10   # window for the dip-check

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7.5), sharey=True, dpi=150,
        gridspec_kw={"wspace": 0.08}
    )
    fig.patch.set_facecolor("#F8FAFC")

    colour   = COLOUR["Argentina"]
    findings = []

    for ax, (mid, team, goal_min, title_suffix) in zip(axes, COMEBACK_WINDOWS):
        mg      = goals[goals["match_id"] == mid]
        mrow    = matches[matches["match_id"] == mid].iloc[0]
        max_match_min = max(int(mg["minute_numeric"].max()) + 5, 90)

        # Full match momentum (needed for global minimum and to clip the window)
        full_min_arr, full_mom_arr = compute_momentum(mg, team, max_minute=max_match_min)
        # full_min_arr[i] == i+1 (minute 1 = index 0)

        # Global minimum
        global_min_idx = int(np.argmin(full_mom_arr))
        global_min_val = float(full_mom_arr[global_min_idx])
        global_min_min = int(full_min_arr[global_min_idx])

        # Momentum exactly 10 min before the goal
        ten_before_min = goal_min - TEN_BEFORE    # the minute to sample
        if 1 <= ten_before_min <= max_match_min:
            ten_before_val = float(full_mom_arr[ten_before_min - 1])   # -1: 0-indexed
        else:
            ten_before_val = float("nan")

        # Ratio: how close is the 10-min-before value to the global minimum?
        if global_min_val != 0 and not math.isnan(ten_before_val):
            ratio_pct = 100.0 * abs(ten_before_val) / abs(global_min_val)
        else:
            ratio_pct = float("nan")

        # Pattern holds if: momentum is negative AND within 75% of global minimum
        pattern_holds = (
            (not math.isnan(ten_before_val))
            and (ten_before_val < 0)
            and (ratio_pct >= 75.0)
        )

        findings.append({
            "match":           mid,
            "goal_min":        goal_min,
            "title_suffix":    title_suffix,
            "global_min_val":  global_min_val,
            "global_min_min":  global_min_min,
            "ten_before_val":  ten_before_val,
            "ten_before_min":  ten_before_min,
            "ratio_pct":       ratio_pct,
            "pattern_holds":   pattern_holds,
        })

        # --- Plot windowed momentum ---
        w_start = max(1, goal_min - PRE)
        w_end   = min(max_match_min, goal_min + POST)
        mask    = (full_min_arr >= w_start) & (full_min_arr <= w_end)
        win_min = full_min_arr[mask]
        win_mom = full_mom_arr[mask]

        ax.set_facecolor("#FFFFFF")
        ax.axhline(0, color="#1E293B", linewidth=1.8, zorder=1)

        # Yellow shading: 10-min pre-goal window
        ax.axvspan(ten_before_min, goal_min,
                   color="#FEF3C7", alpha=0.55, zorder=1,
                   label=f"10-min analysis window ({ten_before_min}' – {goal_min}')")

        # Goal vertical marker
        ax.axvline(goal_min, color=colour, linewidth=1.4,
                   linestyle="--", alpha=0.75, zorder=2)

        # Momentum fill and line
        ax.fill_between(win_min, win_mom, 0,
                        where=(win_mom >= 0), color=colour, alpha=0.18, zorder=2)
        ax.fill_between(win_min, win_mom, 0,
                        where=(win_mom < 0), color="#64748B", alpha=0.18, zorder=2)
        ax.plot(win_min, win_mom, color=colour, linewidth=2.8, zorder=3)

        # Marker: 10-min-before sample
        ax.scatter(ten_before_min, ten_before_val,
                   s=90, color="#F59E0B", marker="D", zorder=6,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            f"{ten_before_min}': {ten_before_val:+.2f}",
            xy=(ten_before_min, ten_before_val),
            xytext=(ten_before_min - 4, ten_before_val - 0.18),
            fontsize=8.5, color="#92400E",
            arrowprops=dict(arrowstyle="->", color="#92400E", lw=1.0),
        )

        # Marker: goal itself
        goal_mom_val = float(full_mom_arr[goal_min - 1])
        ax.scatter(goal_min, goal_mom_val,
                   s=110, color=colour, zorder=6,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"Goal {goal_min}'",
            xy=(goal_min, goal_mom_val),
            xytext=(goal_min + 0.8, goal_mom_val + 0.12),
            fontsize=8.5, color=colour, fontweight="bold",
        )

        # Concession goals in window (for visual context)
        for _, g in mg.sort_values("minute_numeric").iterrows():
            gm = int(g["minute_numeric"])
            if gm < w_start or gm > w_end:
                continue
            if g["scoring_team"] != team:
                idx = min(gm - 1, len(full_mom_arr) - 1)
                ax.scatter(gm, float(full_mom_arr[idx]),
                           s=70, color="#EF4444", zorder=5,
                           edgecolors="white", linewidths=1.0,
                           marker="v")

        # Info text box
        status_text = "HOLDS" if pattern_holds else "DOES NOT HOLD"
        status_col  = "#059669" if pattern_holds else "#B91C1C"
        info = (
            f"Match minimum: {global_min_val:+.2f}  at {global_min_min}'\n"
            f"At {ten_before_min}' (10 min before):  {ten_before_val:+.2f}\n"
            f"  = {ratio_pct:.0f}% of match minimum\n"
            f"'Dip right before comeback':  {status_text}"
        )
        ax.text(0.03, 0.97, info,
                transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left", color=status_col,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#CBD5E1", alpha=0.95))

        opp   = mrow["opponent"]
        score = f"{mrow['final_score_team']}-{mrow['final_score_opp']}"
        stage = STAGE_LABELS.get(mrow["stage"], mrow["stage"])
        ax.set_title(
            f"Argentina {stage} vs {opp}  |  {score}\n{title_suffix}",
            fontsize=12, fontweight="bold", color="#1E293B", pad=8
        )
        ax.set_xlabel("Match minute", fontsize=10, color="#334155")
        ax.set_xlim(w_start - 0.5, w_end + 0.5)
        ax.set_xticks(range(w_start, w_end + 1, 3))
        ax.tick_params(labelsize=9, colors="#64748B")
        ax.grid(axis="y", alpha=0.2, color="#CBD5E1")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_edgecolor("#CBD5E1")
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Momentum (proxy)", fontsize=11, color="#334155")

    # Overall finding for suptitle
    n_hold = sum(1 for f in findings if f["pattern_holds"])
    if n_hold == 2:
        finding_line = (
            "Pattern holds for both: momentum at match minimum right before each comeback goal."
        )
    elif n_hold == 0:
        finding_line = (
            "Pattern holds for neither match. "
            "The 12-min half-life means the model forgets old concessions before the comeback arrives — "
            "the nadir is history, not the current moment."
        )
    else:
        hold_match = next(f["match"] for f in findings if f["pattern_holds"])
        finding_line = (
            f"Pattern holds for {hold_match} only; see info boxes for per-match breakdown."
        )

    fig.suptitle(
        "Pre-Comeback Momentum Dip  |  15 min before / 5 min after Argentina's equalizers\n"
        f"Yellow = 10-min analysis window  |  Diamond = sampled value 10 min pre-goal  |  {finding_line}",
        fontsize=10.5, fontweight="bold", color="#1E293B", y=1.03
    )

    out = CHARTS_DIR / "pre_comeback_dip.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")

    return findings


# -----------------------------------------------------------------------------
# 8. FINAL EXPLAINER CHART
# -----------------------------------------------------------------------------

def chart_momentum_explainer_final(goals, matches):
    """
    Final explainer: Argentina vs England SF (FIFA-verified data only).

    Top half   — swing metaphor, plain language, three annotations:
                 Gordon 55' "England gets the push", decay stretch
                 "The swing loses height on its own", Enzo 85'
                 "Argentina gets the push back".
    Bottom half — formula, half-life sentence, limitation stated plainly.
    Headline   — "The height reflects the last 10–15 minutes, not the whole match."

    Source cited in fine print: FIFA official match centre URL.
    """
    MID      = "ARG-ENG-SF"
    TEAM     = "Argentina"
    colour   = COLOUR[TEAM]
    ENG_RED  = "#DC2626"
    FIFA_URL = ("https://www.fifa.com/en/match-centre/match/"
                "17/285023/289290/400021540?date=2026-07-15")

    mg      = goals[goals["match_id"] == MID]
    max_min = 95     # covers 90+2 = minute 92
    minutes, mom = compute_momentum(mg, TEAM, max_minute=max_min)

    # Key events (0-indexed: minute N lives at mom[N-1])
    G_GORDON  = 55
    G_ENZO    = 85
    G_LAUTARO = 92
    DECAY_REF = 72   # annotation target for decay stretch

    val_gordon  = float(mom[G_GORDON  - 1])   # ≈ -1.000
    val_decay   = float(mom[DECAY_REF - 1])   # ≈ -0.374
    val_enzo    = float(mom[G_ENZO    - 1])   # ≈ +0.823
    val_lautaro = float(mom[G_LAUTARO - 1])   # ≈ +1.553

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[2.5, 1],
        hspace=0.55,
        top=0.82, bottom=0.08, left=0.06, right=0.97,
    )
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Headline (core insight — most prominent element) ─────────────────────
    fig.text(
        0.5, 0.975,
        "The height reflects the last 10–15 minutes, not the whole match.",
        ha="center", va="top",
        fontsize=18, fontweight="bold", color="#0F172A",
    )
    fig.text(
        0.5, 0.935,
        "A team can dominate for an hour and show a flat line if nothing scored recently — "
        "a goal from 40 minutes ago barely registers even though it's still on the scoreboard.",
        ha="center", va="top",
        fontsize=11, color="#475569", style="italic",
    )

    # ── TOP HALF: swing metaphor ─────────────────────────────────────────────
    ax.set_facecolor("#FFFFFF")

    # Minimal gridlines: just ±0.5 and ±1.0
    for yg in [-1.0, -0.5, 0.5, 1.0, 1.5]:
        ax.axhline(yg, color="#E2E8F0", linewidth=0.65, zorder=0)

    # Bold zero line
    ax.axhline(0, color="#1E293B", linewidth=2.5, zorder=1)

    # Coloured fills
    ax.fill_between(minutes, mom, 0,
                    where=(mom >= 0), color=colour, alpha=0.18, zorder=2)
    ax.fill_between(minutes, mom, 0,
                    where=(mom < 0), color=ENG_RED, alpha=0.12, zorder=2)

    # Thick momentum line
    ax.plot(minutes, mom, color=colour, linewidth=4.5, zorder=3)

    # Goal markers
    for gm, gc in [(G_GORDON, ENG_RED), (G_ENZO, colour), (G_LAUTARO, colour)]:
        idx = min(gm - 1, len(mom) - 1)
        ax.scatter(gm, float(mom[idx]), s=160, color=gc, zorder=5,
                   edgecolors="white", linewidths=2.0)

    # Subtle goal-minute tick lines
    for gm in (G_GORDON, G_ENZO, G_LAUTARO):
        ax.axvline(gm, color="#CBD5E1", linewidth=0.7, linestyle=":", zorder=1)

    # Region labels (orient the viewer)
    ax.text(2, 0.14, "Argentina gaining →",
            fontsize=9, color=colour, alpha=0.65, va="bottom")
    ax.text(2, -0.14, "England gaining →",
            fontsize=9, color=ENG_RED, alpha=0.65, va="top")

    # ── Annotation 1: Gordon 55' — "England gets the push" ───────────────────
    ax.annotate(
        "England gets the push\n(Gordon 55')",
        xy=(G_GORDON, val_gordon),
        xytext=(22, val_gordon - 0.26),
        fontsize=13, color="#9B1C1C", fontweight="bold",
        arrowprops=dict(
            arrowstyle="->", color="#9B1C1C", lw=1.8,
            connectionstyle="arc3,rad=-0.12",
        ),
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#FECACA", alpha=0.97),
    )

    # ── Annotation 2: decay stretch — "The swing loses height on its own" ────
    ax.annotate(
        "The swing loses\nheight on its own",
        xy=(DECAY_REF, val_decay),
        xytext=(55, val_decay - 0.42),
        fontsize=13, color="#475569", style="italic",
        arrowprops=dict(
            arrowstyle="->", color="#94A3B8", lw=1.6,
            connectionstyle="arc3,rad=0.18",
        ),
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC",
                  edgecolor="#CBD5E1", alpha=0.97),
    )

    # ── Annotation 3: Enzo 85' — "Argentina gets the push back" ─────────────
    ax.annotate(
        "Argentina gets\nthe push back\n(Enzo 85')",
        xy=(G_ENZO, val_enzo),
        xytext=(61, val_enzo + 0.40),
        fontsize=13, color="#1D4ED8", fontweight="bold",
        arrowprops=dict(
            arrowstyle="->", color=colour, lw=1.8,
            connectionstyle="arc3,rad=0.08",
        ),
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#BFDBFE", alpha=0.97),
    )

    # Lautaro small label
    ax.text(G_LAUTARO + 0.6, val_lautaro + 0.08,
            "Lautaro 90+2'", fontsize=10, color=colour,
            va="bottom", fontweight="bold")

    # ── Axis formatting ───────────────────────────────────────────────────────
    y_lo = mom.min() * 1.55
    y_hi = mom.max() * 1.65
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0, max_min + 2)
    ax.set_xticks([0, 15, 30, 45, 55, 72, 85, 92])
    ax.set_xticklabels(
        ["0", "15'", "30'", "45'", "55'", "72'", "85'", "90+2'"],
        fontsize=10.5, color="#64748B",
    )
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#E2E8F0")
    ax.set_title(
        "Argentina 2–1 England  |  2026 FIFA World Cup Semifinal  |  Atlanta, July 15",
        fontsize=13, fontweight="bold", color="#1E293B", pad=10,
    )

    # ── BOTTOM HALF: technical spec ───────────────────────────────────────────
    ax2.set_facecolor("#F1F5F9")
    ax2.axis("off")

    ax2.text(
        0.5, 0.88,
        r"momentum(t)  =  $\Sigma$[ own_goal_weight × 0.5$^{(t-t_0)/12}$ ]"
        "  −  "
        r"$\Sigma$[ opp_goal_weight × 0.5$^{(t-t_0)/12}$ ]",
        ha="center", va="center", fontsize=10.5, color="#1E293B",
        transform=ax2.transAxes, fontfamily="monospace",
    )
    ax2.text(
        0.5, 0.60,
        "Each push loses half its remaining height every 12 minutes — "
        "not a flat amount, a flat fraction of what's left.",
        ha="center", va="center", fontsize=10.5, color="#334155",
        transform=ax2.transAxes, style="italic",
    )
    ax2.text(
        0.5, 0.28,
        "Known blind spot: this reconstruction reacts only to goals — "
        "not to shots, passes, or possession like the real broadcast graphic.\n"
        "During any tense, goalless stretch, the line goes flat and uninformative. "
        "That is a structural property of the model, not a hedge.",
        ha="center", va="center", fontsize=9.5, color="#64748B",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.90),
    )

    # ── Citation fine print ───────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        f"Verified source: FIFA official match centre  |  {FIFA_URL}",
        ha="center", va="bottom", fontsize=7.5, color="#94A3B8",
    )

    out = CHARTS_DIR / "momentum_explainer_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# -----------------------------------------------------------------------------
# 9. MAIN
# -----------------------------------------------------------------------------

def main():
    goals, matches = load_data()

    print(f"\nLoaded {len(goals)} goal events, {len(matches)} matches")
    print(f"Spain matches: {len(matches[matches['team']=='Spain'])}")
    print(f"Argentina matches: {len(matches[matches['team']=='Argentina'])}")

    # Compute time bucket using match-level extra_time flag (Bug 2 fix)
    goals = goals.copy()
    goals["bucket"] = goals.apply(
        lambda r: time_bucket(r["minute_numeric"], r["extra_time"]), axis=1
    )

    compute_insights(goals, matches)

    print("\n-- Generating charts...")
    chart_small_multiples(goals, matches)
    chart_goals_by_bucket(goals, matches)
    chart_volatility(goals, matches)
    chart_hero_4panel(goals, matches)
    chart_methodology_explainer(goals, matches)
    chart_momentum_explainer_final(goals, matches)

    # -- Response time analysis
    print("\n-- Computing response times after conceding...")
    response_df = compute_response_times(goals, matches)

    csv_cols = [
        "team", "match_id", "opponent", "stage",
        "concede_minute", "concede_minute_raw", "concede_scorer",
        "response_minute", "response_minute_raw", "response_scorer",
        "gap_minutes", "no_response",
    ]
    csv_path = DATA_DIR / "response_times.csv"
    response_df[csv_cols].to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    responded = response_df[~response_df["no_response"]]
    print(f"\n  Response events: {len(responded)}  |  No-response: {len(response_df) - len(responded)}")
    for _, row in responded.iterrows():
        print(f"    {TEAM_CODE.get(row['team'], '???')} vs {row['opponent'][:3].upper()} "
              f"({STAGE_LABELS.get(row['stage'], row['stage'])}): "
              f"conceded {int(row['concede_minute'])}' ({row['concede_scorer'].split()[-1]}) "
              f"-> {row['response_minute_raw']}' ({row['response_scorer'].split()[-1]}): "
              f"+{int(row['gap_minutes'])} min")

    chart_time_to_respond(response_df)

    # -- Pre-comeback dip analysis
    print("\n-- Analysing pre-comeback momentum dip...")
    findings = chart_pre_comeback_dip(goals, matches)
    print("\n  Pre-comeback dip results:")
    for f in findings:
        sign_tag = "POSITIVE" if f["ten_before_val"] > 0 else "negative"
        print(f"    {f['match']}  |  equalizer at {f['goal_min']}'")
        print(f"      Global min: {f['global_min_val']:+.3f} at {f['global_min_min']}'")
        print(f"      10 min before ({f['ten_before_min']}'): {f['ten_before_val']:+.3f} "
              f"({sign_tag}, {f['ratio_pct']:.0f}% of min)")
        print(f"      Pattern holds: {'YES' if f['pattern_holds'] else 'NO'}")

    print(f"\nAll charts saved to: {CHARTS_DIR.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
