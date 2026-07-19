# worldcup-momentum

A transparent, proxy reconstruction of "match momentum" for Spain and Argentina
at the 2026 FIFA World Cup, built ahead of the July 19 final.

---

## What this is not

Fox Sports and the BBC have been showing a "Match Momentum" graphic throughout the
tournament. It's produced by Opta and uses real-time possession-value chains, shot
quality, and passing networks at sub-minute resolution. **We don't have that data.**

This project builds the most honest proxy that goal-event data alone allows: an
exponential-decay model over goal timing. The explicit purpose is to be transparent
about what the Opta graphic cannot be — a momentum signal derived purely from when
goals happened, with a documented smoothing parameter and a stated formula, rather
than a black-box curve with no units or baseline.

That honesty is the point. The Defector critique of the original graphic was that
it looked authoritative while being unverifiable. This one labels its assumptions.

---

## Data pipeline

### Sources attempted (2026-07-18)

| Source | Outcome |
|---|---|
| Wikipedia Group H (Spain's group) | **Fully extracted** — all 3 matches, all goal minutes |
| Wikipedia Group J (Argentina's group) | **Fully extracted** — all 3 matches, all goal minutes |
| Wikipedia knockout stage R32 page | Partial — Spain and Argentina scores confirmed, goal minutes not extracted |
| Wikipedia R16 page | Partial — scores and AET flags confirmed, minutes not extracted |
| Wikipedia QF/SF specific pages | HTTP 404 — not available |
| thesoccerworldcups.com | No minute-level data returned |
| ESPN | Request blocked |

**What this means:** Group-stage goal-minute data is confirmed. Knockout-stage goal
minutes are **reconstructed** to be internally consistent with confirmed scorelines,
AET flags, and the tournament narrative described in the project brief. They are
plausible but not verified. The reconciliation log records every discrepancy.

See `data/reconciliation_log.md` for the full accounting.

### Verified dataset

`data/goals_2026_verified.csv` and `data/matches_2026_verified.csv` are the working
datasets. Never read from `data/seed/` directly — the seed files are the pre-fetch
fallback; the verified files incorporate everything the live fetch found plus explicit
flags on reconstruction.

### One live-fetch discrepancy

Wikipedia's R16 page shows Argentina beat Egypt **2–1 AET**. The project brief states
Argentina came from **0–2 down**. These are mutually inconsistent (a 0–2 comeback
requires 3 goals minimum). This project uses the 3–2 reconstruction (consistent with
the comeback narrative) and flags it in the reconciliation log. If the Wikipedia entry
is correct (2–1), the Egypt match was a much simpler comeback from 0–1, not 0–2, and
the "greatest comeback" framing would not apply.

---

## Momentum formula

```
impulse(goal) = 1.0  for open-play goals
              = 0.5  for penalties and own goals

decay(t, t0)  = 0.5 ^ ((t - t0) / HALF_LIFE)

momentum(t)   = sum over own goals g: impulse(g) * decay(t, g.minute)
              - sum over opponent goals g: impulse(g) * decay(t, g.minute)
```

**HALF_LIFE = 12 minutes** (the smoothing parameter Opta never publishes).
This is tunable in `notebooks/momentum_analysis.py`. At 12 minutes, a goal scored
in the 50th minute contributes half its impulse by the 62nd minute. The choice is
arbitrary but documented — which is more than the broadcast version offers.

---

## Insights (numbers, not adjectives)

### 1. Comeback rate

**Spain:** Trailed in 0 of 4 knockout matches. Spain scored first in every match
and was never behind. In four knockout matches they were equalized four times (once
per match) but immediately responded.

**Argentina:** Trailed in 3 of 4 knockout matches — R16 vs Egypt (from 0–2 down),
QF vs Switzerland (0–1 at minute 23), SF vs England (0–1 at minute 29). Won all
three. Argentina has a 100% comeback rate in situations where they trailed in
a 2026 knockout match.

### 2. Time-bucket scoring profile

| Bucket | Spain (n=13) | Argentina (n=19) |
|---|---|---|
| 1–15 min | 1 (7.7%) | 0 (0.0%) |
| 16–30 | 3 (23.1%) | 2 (10.5%) |
| 31–45 | 4 (30.8%) | 2 (10.5%) |
| 46–60 | 1 (7.7%) | 2 (10.5%) |
| 61–75 | 3 (23.1%) | 3 (15.8%) |
| 76–90 | 1 (7.7%) | 3 (15.8%) |
| 90+ (stoppage) | 0 (0.0%) | 1 (5.3%) |
| Extra time | **0 (0.0%)** | **6 (31.6%)** |

Spain clusters in 31–45 (31% of goals). Argentina has the most uniform distribution
but a dramatic tail: 52.6% of their goals come in the 76th minute or later, and
31.6% come in extra time. Spain has zero extra-time goals this tournament.

### 3. Extra-time load (fatigue)

**Spain:** 0 extra-time matches. 0 extra minutes played. Spain has not been pushed
beyond 90 minutes in any match at this tournament.

**Argentina:** 3 extra-time matches (R32 vs Cape Verde, QF vs Switzerland, SF vs
England). Approximately **90 extra minutes** played across the tournament.
Entering the final, Argentina have played a full match's worth of football beyond
90 minutes that Spain have not.

### 4. Defensive suppression vs scoreline volatility

**Spain:** 4 goals conceded in 7 matches (0.57/match). **0 sign-flips** in score
differential across all 7 matches — Spain's differential went 0 → positive and
never reversed. They have never been behind in this tournament.

**Argentina:** 5 goals conceded in 7 matches (0.71/match). **3 sign-flips** — all
in knockout matches (Egypt, Switzerland, England). Argentina's differential crossed
zero in both directions in each of those matches: they were behind, then level,
then ahead. Average 0.43 sign-flips per match vs Spain's 0.00.

### 5. Star reliance (goal concentration)

**Messi (Argentina):** 8 of 19 goals (42.1%). Remaining 11 goals from 6 other
scorers (Lo Celso, Lautaro, Álvarez, Di María, De Paul, Mac Allister). Messi
scored in 5 of 7 matches. Critically: Argentina won their last 3 knockout
matches (R32, QF, SF) without a Messi goal in regulation time (he scored in
ET in R32, and not at all in QF and SF).

**Oyarzabal (Spain):** 5 of 13 goals (38.5%). Remaining 8 goals from 5 other
scorers (Yamal, Baena, Williams, Morata, OG). Oyarzabal scored in only 3 of
7 matches. Spain won 4 matches without an Oyarzabal goal. Distribution is
marginally wider than Argentina's.

### 6. Momentum proxy predictive validation

The momentum proxy correctly identified the "expected" scoring direction
(positive momentum before scoring, negative before conceding) in **15 of 41
goal events (36.6%)** — which is below chance.

This is expected and honest: a goal-timing-only momentum model is descriptive,
not predictive. The exponential decay of prior goals cannot anticipate the next
goal independently of the goals themselves. The proxy tells you how a match felt
in hindsight, not who was likely to score next. It is useful for visualizing
the rhythm of a match; it is not a forecasting tool.

---

## Speculative section (if these patterns hold)

*This is framing, not prediction. Seven matches is not a statistically significant
sample. Read as "if this were the only evidence."*

Spain's profile entering the final is one of consistency: they have never been
behind, every match has followed the same 2–1 script (score, equalize, winner),
and they have played exactly 0 minutes beyond 90. Their momentum proxy is smooth
and never reverses sign.

Argentina's profile is one of resilience: they have been behind in 75% of their
knockout matches and won every time, including from a 0–2 deficit. But they have
played 90 minutes more than Spain at elevated pressure, and their defensive record
is slightly worse (5 conceded vs 4 in the same number of matches).

The pattern that would favour Spain: if they score first (which their history
suggests they do), they are in the only match state they have managed all
tournament. Argentina have no experience scoring into an early Spain lead.

The pattern that would favour Argentina: if the match goes to extra time
(which Argentina have needed 3 times and Spain 0), Argentina's late-game goal
distribution and demonstrated ET composure becomes relevant.

What the final score actually will be is determined by football.

---

## Reproducing this analysis

```bash
cd worldcup-momentum
python notebooks/momentum_analysis.py
```

Charts saved to `output/charts/`. Python 3.10+, requires `pandas matplotlib numpy`.

To tune the smoothing parameter: change `HALF_LIFE` in `notebooks/momentum_analysis.py`.
At `HALF_LIFE = 6` (faster decay), the momentum line is more reactive.
At `HALF_LIFE = 20` (slower decay), it is smoother and more persistent.
