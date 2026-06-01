"""
Colombia 2026 Presidential Election — Multi-Agent Simulator
============================================================
Four-agent pipeline:
  1. Policy Analyst   — maps candidates across 5 policy dimensions
  2. Voter Bloc x5    — one agent per congressional bloc
  3. Median Voter     — applies Median Voter Theorem
  4. Orchestrator     — synthesizes all findings

Usage:
    export ANTHROPIC_API_KEY=your_key
    python simulator.py
"""

import json
from pathlib import Path

import anthropic

from data import CANDIDATES_2026, CONGRESSIONAL_RESULTS_2026, POLICY_DIMENSIONS
from prompts import (
    MEDIAN_VOTER_PROMPT,
    ORCHESTRATOR_PROMPT,
    POLICY_ANALYST_PROMPT,
    VOTER_BLOC_PROMPT,
)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


# ── Shared helper ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Parse JSON from model output, stripping markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence line; drop closing ``` if present
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        joined = "\n".join(inner)
        if joined.lstrip().startswith("json"):
            joined = joined.lstrip()[4:]
        return json.loads(joined.strip())
    return json.loads(text)


# ── Agent 1: Policy Analyst ──────────────────────────────────────────────────

def run_policy_analyst(candidates: dict, dimensions: list) -> dict:
    """
    Ask Claude to map all candidates across all policy dimensions.
    Returns structured JSON with policy_map, policy_gaps, consensus_areas.
    """
    print("\n" + "─" * 70)
    print("AGENT 1: Policy Analyst")
    print("─" * 70)

    user_message = f"""Analyze the following Colombian presidential candidates across all policy dimensions.

CANDIDATES AND PROPOSALS:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

POLICY DIMENSIONS TO ANALYZE: {dimensions}

Return ONLY a JSON object with this exact structure:
{{
  "policy_map": {{
    "<dimension>": {{
      "<candidate_name>": {{
        "position_summary": "<2-3 sentence summary>",
        "ideology_score": <float -1 to 1>,
        "key_proposal": "<the single most defining proposal>"
      }}
    }}
  }},
  "policy_gaps": [
    {{
      "dimension": "<dimension name>",
      "gap_description": "<what the disagreement is about>",
      "left_position": "<what the left candidate proposes>",
      "right_position": "<what the right candidate proposes>",
      "gap_magnitude": <float 0 to 1>
    }}
  ],
  "consensus_areas": ["<area of agreement>"]
}}

No preamble. No explanation. Only valid JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=POLICY_ANALYST_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    result = _parse_json(response.content[0].text)

    # ── Print: ideology score table ──────────────────────────────────────────
    cand_names = list(candidates.keys())
    col_w = 18
    print(f"\n  {'Dimension':<13}", end="")
    for name in cand_names:
        print(f"  {name[:col_w - 2]:<{col_w}}", end="")
    print()
    print("  " + "─" * (13 + (col_w + 2) * len(cand_names)))

    for dim in dimensions:
        print(f"  {dim:<13}", end="")
        dim_data = result.get("policy_map", {}).get(dim, {})
        for name in cand_names:
            score = dim_data.get(name, {}).get("ideology_score", 0.0)
            cell = f"{score:+.2f}"
            print(f"  {cell:<{col_w}}", end="")
        print()

    # ── Print: top 3 policy gaps ─────────────────────────────────────────────
    print("\n🔺 Top Policy Gaps:")
    gaps = sorted(
        result.get("policy_gaps", []),
        key=lambda g: g.get("gap_magnitude", 0.0),
        reverse=True,
    )[:3]
    for i, gap in enumerate(gaps, 1):
        mag = gap.get("gap_magnitude", 0.0)
        print(f"  {i}. {gap.get('dimension', '').upper()} — magnitude: {mag:.2f}")
        print(f"     {gap.get('gap_description', '')}")

    # ── Print: consensus areas ───────────────────────────────────────────────
    print("\n✅ Areas of Consensus:")
    for area in result.get("consensus_areas", []):
        print(f"  · {area}")

    return result


# ── Agent 2: Voter Bloc Agents ───────────────────────────────────────────────

def _run_single_bloc(bloc_name: str, bloc_data: dict, candidates: dict) -> dict:
    """Call the voter bloc agent for one congressional bloc."""
    candidate_summary = {
        name: {
            "ideology_score": cdata["ideology_score"],
            "coalition": cdata["coalition"],
            "key_proposals": cdata["key_proposals"],
        }
        for name, cdata in candidates.items()
    }

    user_message = f"""Model the voting preferences of this Colombian congressional bloc.

BLOC: {bloc_name}
SEATS IN SENATE: {bloc_data.get('seats_senate', 0)}
IDEOLOGY SCORE: {bloc_data.get('ideology_score', 0)} (-1 far left, +1 far right)
CURRENT PRESIDENTIAL CANDIDATE: {bloc_data.get('presidential_candidate', 'TBD')}

CANDIDATES AND POSITIONS:
{json.dumps(candidate_summary, ensure_ascii=False, indent=2)}

Return ONLY a JSON object with this exact structure:
{{
  "bloc": "{bloc_name}",
  "seats": {bloc_data.get('seats_senate', 0)},
  "ideology_score": {bloc_data.get('ideology_score', 0)},
  "round_1_preference": "<candidate name>",
  "round_1_confidence": <float 0 to 1>,
  "round_2_scenarios": {{
    "Valencia_vs_Cepeda": {{
      "preference": "<candidate name or 'abstain'>",
      "probability": <float 0 to 1>,
      "reasoning": "<1-2 sentences>"
    }},
    "Valencia_vs_Lopez": {{
      "preference": "<candidate name>",
      "probability": <float 0 to 1>,
      "reasoning": "<1-2 sentences>"
    }},
    "Cepeda_vs_Lopez": {{
      "preference": "<candidate name>",
      "probability": <float 0 to 1>,
      "reasoning": "<1-2 sentences>"
    }}
  }}
}}

No preamble. Only valid JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=VOTER_BLOC_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return _parse_json(response.content[0].text)


def run_voter_blocs(blocs: dict, candidates: dict) -> list:
    """Run one voter bloc agent per congressional bloc, sequentially."""
    print("\n" + "─" * 70)
    print("AGENT 2: Voter Bloc Agents (5 blocs)")
    print("─" * 70)

    results = []
    for bloc_name, bloc_data in blocs.items():
        seats = bloc_data["seats_senate"]
        score = bloc_data["ideology_score"]
        print(f"\n  Analyzing: {bloc_name}  ({seats} seats · ideology {score:+.2f})")

        result = _run_single_bloc(bloc_name, bloc_data, candidates)
        results.append(result)

        r1 = result.get("round_1_preference", "?")
        conf = result.get("round_1_confidence", 0.0)
        print(f"    Round 1 → {r1}  (confidence {conf:.0%})")

        for matchup, data in result.get("round_2_scenarios", {}).items():
            label = matchup.replace("_vs_", " vs ")
            pref = data.get("preference", "?")
            prob = data.get("probability", 0.0)
            print(f"    {label:<30}  → {pref}  ({prob:.0%})")

    return results


# ── Agent 3: Median Voter ────────────────────────────────────────────────────

def run_median_voter_agent(bloc_results: list, candidates: dict) -> dict:
    """Apply the Median Voter Theorem using bloc seat counts as voter weights."""
    print("\n" + "─" * 70)
    print("AGENT 3: Median Voter Agent — Median Voter Theorem")
    print("─" * 70)

    candidate_positions = {
        name: {"ideology_score": cdata["ideology_score"], "coalition": cdata["coalition"]}
        for name, cdata in candidates.items()
    }

    user_message = f"""Apply the Median Voter Theorem to Colombia's 2026 presidential election.

VOTER BLOC RESULTS (seats = voter weight):
{json.dumps(bloc_results, ensure_ascii=False, indent=2)}

CANDIDATE IDEOLOGY POSITIONS:
{json.dumps(candidate_positions, ensure_ascii=False, indent=2)}

Instructions:
1. Calculate the seat-weighted median ideology position across all blocs
2. Identify which candidate sits closest to that median
3. Model all three second-round scenarios
4. Identify the most likely second-round matchup based on round 1 bloc preferences

Return ONLY a JSON object with this exact structure:
{{
  "median_voter_position": <float -1 to 1>,
  "median_voter_description": "<describe who this median voter is in Colombian political terms>",
  "second_round_scenarios": [
    {{
      "matchup": "<Candidate A vs Candidate B>",
      "median_voter_closest_to": "<candidate name>",
      "predicted_winner": "<candidate name>",
      "win_probability": <float 0.5 to 1.0>,
      "key_factor": "<the decisive political or structural factor>"
    }}
  ],
  "most_likely_second_round": "<most likely matchup>",
  "analysis": "<2-3 paragraph analysis of second round dynamics>"
}}

No preamble. Only valid JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=MEDIAN_VOTER_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    result = _parse_json(response.content[0].text)

    pos = result.get("median_voter_position", 0.0)
    desc = result.get("median_voter_description", "")
    print(f"\n📍 Median Voter Position: {pos:+.3f}")
    print(f"   {desc}")

    print("\n⚖️  Second Round Scenarios:")
    for scenario in result.get("second_round_scenarios", []):
        matchup = scenario.get("matchup", "?")
        winner = scenario.get("predicted_winner", "?")
        prob = scenario.get("win_probability", 0.0)
        closest = scenario.get("median_voter_closest_to", "?")
        factor = scenario.get("key_factor", "")
        print(f"\n  {matchup}")
        print(f"    Median voter closest to : {closest}")
        print(f"    Predicted winner        : {winner}  ({prob:.0%})")
        print(f"    Key factor              : {factor}")

    likely = result.get("most_likely_second_round", "?")
    print(f"\n🎯 Most Likely Second Round: {likely}")

    return result


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_orchestrator() -> dict:
    """
    Coordinate all three agents in sequence, then produce a final
    neutral synthesis of the second-round dynamics.
    """
    all_results: dict = {}

    # ── Agent 1 ──────────────────────────────────────────────────────────────
    policy_result = run_policy_analyst(CANDIDATES_2026, POLICY_DIMENSIONS)
    all_results["policy_analysis"] = policy_result

    # ── Agent 2 ──────────────────────────────────────────────────────────────
    bloc_results = run_voter_blocs(CONGRESSIONAL_RESULTS_2026, CANDIDATES_2026)
    all_results["voter_bloc_results"] = bloc_results

    # ── Agent 3 ──────────────────────────────────────────────────────────────
    median_result = run_median_voter_agent(bloc_results, CANDIDATES_2026)
    all_results["median_voter_analysis"] = median_result

    # ── Final synthesis ───────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("ORCHESTRATOR: Final Synthesis")
    print("─" * 70)

    gaps = policy_result.get("policy_gaps", [])
    top_gap_dim = gaps[0]["dimension"].upper() if gaps else "N/A"
    consensus = policy_result.get("consensus_areas", [])

    bloc_summary = "\n".join(
        f"  - {r['bloc']} ({r['seats']} seats): "
        f"Round 1 → {r.get('round_1_preference', '?')}  "
        f"(confidence {r.get('round_1_confidence', 0):.0%})"
        for r in bloc_results
    )

    synthesis_prompt = f"""Given the following analysis of Colombia's 2026 presidential election:

POLICY ANALYSIS:
- Policy gaps identified: {len(gaps)} — largest gap dimension: {top_gap_dim}
- Areas of cross-candidate consensus: {consensus}

VOTER BLOC ROUND 1 PREFERENCES:
{bloc_summary}

MEDIAN VOTER:
- Seat-weighted position: {median_result.get('median_voter_position', 0):+.3f}
- Description: {median_result.get('median_voter_description', '')}
- Most likely second round: {median_result.get('most_likely_second_round', '?')}

MEDIAN VOTER ANALYSIS:
{median_result.get('analysis', '')}

Provide a 3-paragraph neutral synthesis of what Colombia's second round dynamics look like \
and which policy dimensions will define the outcome."""

    synthesis_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=ORCHESTRATOR_PROMPT,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesis_text = synthesis_response.content[0].text.strip()
    print(f"\n{synthesis_text}")
    all_results["synthesis"] = synthesis_text

    # ── Save results ─────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("COLOMBIA 2026 PRESIDENTIAL ELECTION — MULTI-AGENT ANALYSIS")
    print("Based on March 8, 2026 congressional results")
    print("=" * 70)

    run_orchestrator()

    print("\n" + "=" * 70)
    print("All results saved to output/simulation_results.json")
    print("=" * 70)
