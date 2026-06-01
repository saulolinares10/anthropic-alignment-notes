# Colombia 2026 Presidential Election Simulator

A multi-agent system that analyzes Colombia's 2026 presidential election using the Anthropic Claude API.

Built the day after Colombia's March 8, 2026 congressional elections.

---

## Architecture

```
run_orchestrator()
│
├── Agent 1: Policy Analyst
│   Reads all candidate proposals, maps 5 policy dimensions,
│   identifies gaps and consensus areas.
│
├── Agent 2: Voter Bloc Agents (×5)
│   One agent per congressional bloc. Each models round 1
│   preference and three possible second-round scenarios.
│
├── Agent 3: Median Voter Agent
│   Takes all bloc results + candidate positions. Applies the
│   Median Voter Theorem using seat counts as voter weights.
│   Predicts second-round winner for each matchup.
│
└── Orchestrator Synthesis
    Receives output from all three agents. Produces a 3-paragraph
    neutral synthesis of second-round dynamics.
```

Each agent is a separate `client.messages.create()` call with its own system prompt — no shared state, no shared context window.

---

## Files

| File | Purpose |
|---|---|
| `simulator.py` | Main orchestration pipeline — all four agents |
| `data.py` | Congressional results and candidate proposals as Python dicts |
| `prompts.py` | System prompts for each agent |
| `output/simulation_results.json` | Full JSON output of a completed run |

---

## Key Question

Given the congressional seat distribution and candidate policy positions — where does the median voter sit on the ideological spectrum, and which second-round matchup favors which candidate?

The March 8 results produced a fragmented congress with no bloc holding more than 36% of seats:

| Bloc | Seats | Ideology |
|---|---|---|
| Pacto Histórico | 25 | -0.80 |
| Centro Democrático | 17 | +0.85 |
| Partido Liberal | 11 | +0.20 |
| Alianza por Colombia | 9 | +0.50 |
| Partido Conservador | 8 | +0.70 |

The seat-weighted median sits in mildly center-right territory — which has significant implications for second-round dynamics in any matchup between the two dominant candidates (Cepeda and Valencia).

---

## Data Sources

- **Congressional results**: Registraduría Nacional del Estado Civil (March 8, 2026)
- **Candidate proposals**: El Tiempo, Razón Pública, Bloomberg Línea, Infobae Colombia (May 2026)

---

## Usage

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
python simulator.py
```

Output is printed to stdout and saved to `output/simulation_results.json`.

---

## Disclaimer

This is an educational simulation. Political prediction is inherently uncertain. This tool is meant to illustrate multi-agent orchestration patterns using the Anthropic Claude API — not to predict election outcomes.

---

## Multi-Agent Pattern Notes

This simulator uses the **sequential orchestration** pattern:
- Each agent has an isolated context (its own system prompt, its own `messages` list)
- The orchestrator passes structured data between agents via Python variables
- No agent has access to another agent's reasoning — only its structured output
- The orchestrator makes a final synthesis call that sees all prior outputs

This is a clean demonstration of how to build a multi-agent pipeline without shared state or inter-agent communication — each agent is stateless and verifiable.
