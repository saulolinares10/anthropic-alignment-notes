# Colombia 2026 · Segunda Vuelta Simulator

A multi-agent system analyzing the June 21st presidential second round between Abelardo de la Espriella and Iván Cepeda.

Built June 1st, 2026 — the day after the first round.

---

## Three specialized agents

| Agent | Job |
|---|---|
| **Survey Analyst** | Reconciles divergent polls from Invamer, AtlasIntel/Semana, and Guarumo/Ecoanalítica — which predict opposite winners |
| **Swing Variable Analyst** | Models Paloma transfer efficiency, Fajardo centrist split, and abstention differential across three scenarios |
| **Median Voter Agent** | Applies MVT using first-round vote shares as weights, then explicitly identifies where Colombian political reality deviates from the theorem |

---

## Why no congressional seats

The first version of this simulator used congressional seat distribution as a proxy for presidential voter preferences. That was methodologically wrong.

Congressional and presidential voting behavior differ significantly in Colombia — the 2026 first round proved this when Abelardo de la Espriella surged to 43.74% despite running without the formal congressional coalition that backed Paloma Valencia.

This version uses only:
- Actual presidential first-round vote counts (Registraduría, May 31 2026)
- Published second-round poll data with sample methodology noted
- 2022 historical second-round patterns as calibration anchor

---

## Data sources

- **Registraduría Nacional del Estado Civil** — May 31, 2026 first round results
- **Invamer / CNN Colombia** — second round poll, May 2026
- **AtlasIntel / Revista Semana** — second round poll, May 2026
- **Guarumo / Ecoanalítica / El Tiempo** — second round poll, May 2026
- **2022 historical data** — Registraduría (second round: Petro 50.44%, Hernández 47.31%)

---

## The polling problem

All three polls were conducted in the same period with access to the same first-round result. Two predict Abelardo wins. One predicts Cepeda wins. The margins differ by 10+ points. This is not a minor methodological disagreement — the Survey Analyst agent is specifically tasked with explaining why, producing a weighted consensus, and identifying what polling structurally cannot capture.

---

## Usage

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
python simulator.py
```

Output is printed to stdout and saved to `output/segunda_vuelta_analysis.json`.

---

## Disclaimer

Electoral prediction is inherently uncertain. This is an educational tool demonstrating multi-agent orchestration on a real political problem with live, conflicting data. The agent's predictions are not electoral forecasts.
