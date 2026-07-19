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

**Spain:** Trailed in 0 of 4 knockout matches. Spain scored first in every knockout
match and was never behind. In all four knockout matches they were equalized (once
per match: Austria R32, Portugal R16, Belgium QF, France SF) but never went behind.

**Argentina:** Trailed in 2 of 4 knockout matches — R16 vs Egypt (from 0–2 down,
68 minutes trailing) and SF vs England (0–1 from minute 55, 30 minutes trailing).
Won both. Argentina has a 100% comeback rate in 2026 knockout matches where they trailed.
**Combined minutes trailing across comeback wins: 98 minutes.**

The QF vs Switzerland (Argentina 3–1 AET) was not a comeback — Argentina scored
first at minute 10' and was equalized at 67', then won in extra time. Same shape
as every Spain knockout match.

### 2. Time-bucket scoring profile

| Bucket | Spain (n=13) | Argentina (n=19) |
|---|---|---|
| 1–15 min | 1 (7.7%) | 1 (5.3%) |
| 16–30 | 3 (23.1%) | 2 (10.5%) |
| 31–45 | 4 (30.8%) | 2 (10.5%) |
| 46–60 | 1 (7.7%) | 2 (10.5%) |
| 61–75 | 2 (15.4%) | 1 (5.3%) |
| 76–90 | 2 (15.4%) | 4 (21.1%) |
| 90+ (stoppage) | 0 (0.0%) | 3 (15.8%) |
| True extra time | **0 (0.0%)** | **4 (21.1%)** |

"True extra time" = match went to AET AND goal minute > 90. Goals at 90+2'
in matches that ended in regulation (Argentina vs Egypt, vs England) are
stoppage time, not extra time. Spain clusters in 31–45 (31% of goals). Argentina's
tail is striking: 57.9% of goals come in the 76th minute or later, including
21.1% in genuine extra time. Spain has zero goals beyond the 90th minute this tournament.

### 3. Extra-time load (fatigue)

**Spain:** 0 extra-time matches. 0 extra minutes played. Spain has not been pushed
beyond 90 minutes in any match at this tournament.

**Argentina:** 2 extra-time matches (R32 vs Cape Verde 3–1 AET, QF vs Switzerland
3–1 AET). Approximately **60 extra minutes** played across the tournament. The SF
vs England ended in stoppage time (Lautaro 90+2'), not extra time — Argentina won
2–1 without requiring the additional 30 minutes.

### 4. Defensive suppression vs scoreline volatility

**Spain:** 4 goals conceded in 7 matches (0.57/match). **0 sign-flips** in score
differential across all 7 matches — Spain's differential went 0 → positive and
never reversed. They have never been behind in this tournament.

**Argentina:** 6 goals conceded in 7 matches (0.86/match). **2 sign-flips** — both
in knockout comeback wins (R16 vs Egypt and SF vs England). Argentina's differential
crossed from negative to positive in exactly those two matches. The Switzerland QF
reads as 0 flips because Argentina led first (Messi 10') and was only equalized,
never behind. Average 0.29 sign-flips per match vs Spain's 0.00.

### 5. Comeback closers

**Argentina's decisive goals in comeback wins:**
- R16 vs Egypt: **Enzo Fernández at 90+2'** (came from 0–2; Di María 56', Messi 83' to equalize, Enzo wins it)
- SF vs England: **Lautaro Martínez at 90+2'** (came from 0–1; Enzo 85' to equalize, Lautaro wins it)

Messi's contribution to the two comebacks: the 83' equalizer vs Egypt only.
Both winning goals were delivered by other players under maximum pressure.

**Goal concentration:**

Messi (Argentina): 8 of 19 goals (42.1%). Remaining 11 goals from 6 other scorers
(Lo Celso, Lautaro, Álvarez, Enzo Fernández, Di María, Mac Allister).

Oyarzabal (Spain): 5 of 13 goals (38.5%). Remaining 8 goals from 5 other scorers
(Yamal, Baena, Fabián Ruiz, Merino, Morata, OG). Spain's QF vs Belgium was won
by Fabián Ruiz (30') and Merino (88') — neither is Oyarzabal.

### 6. Momentum proxy predictive validation

The momentum proxy correctly identified the "expected" scoring direction
(positive momentum before scoring, negative before conceding) in **13 of 42
goal events (31.0%)** — which is below chance.

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

Argentina's profile is one of resilience: they have been behind in 50% of their
knockout matches (Egypt and England) and won both, including from a 0–2 deficit.
They have played 60 extra minutes beyond 90 that Spain have not, and their defensive
record is slightly worse (6 conceded vs 4 in the same number of matches).

The pattern that would favour Spain: if they score first (which their history
suggests they do), they are in the only match state they have managed all
tournament. Argentina have no experience scoring into an early Spain lead.

The pattern that would favour Argentina: if the match goes to extra time
(which Argentina have needed twice and Spain 0), Argentina's late-game goal
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
