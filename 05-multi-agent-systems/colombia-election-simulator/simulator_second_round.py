"""
Colombia 2026 — Segunda Vuelta Simulator
=========================================
Focused second-round analysis: Abelardo de la Espriella vs Iván Cepeda
Date: June 21, 2026

Uses actual May 31st first-round vote percentages as voter weights
instead of congressional seat counts.

Three-agent pipeline:
  1. Voter Bloc Agent (updated)  — models each bloc's second-round alignment
                                   given real first-round results + Valencia endorsement
  2. Median Voter Agent          — recalculates MVT with vote-share weights
  3. Orchestrator synthesis      — Abelardo's path, Cepeda's path, decisive variable

Usage:
    export ANTHROPIC_API_KEY=your_key
    python simulator_second_round.py
"""

import json
from pathlib import Path

import anthropic

from data import (
    CANDIDATES_2026,
    CONGRESSIONAL_RESULTS_2026,
    FIRST_ROUND_RESULTS_2026,
    POLICY_DIMENSIONS,
    SECOND_ROUND_DATE,
    SECOND_ROUND_MATCHUP,
)
from prompts import MEDIAN_VOTER_PROMPT, ORCHESTRATOR_PROMPT, VOTER_BLOC_PROMPT

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# ── Finalists and their first-round data ─────────────────────────────────────

FINALISTS = {
    name: {**CANDIDATES_2026[name], **FIRST_ROUND_RESULTS_2026[name]}
    for name in ("Abelardo de la Espriella", "Iván Cepeda")
}

# Eliminated candidates — their vote shares become the contested pool
ELIMINATED = {
    "Paloma Valencia": {
        **FIRST_ROUND_RESULTS_2026["Paloma Valencia"],
        "ideology_score": CANDIDATES_2026["Paloma Valencia"]["ideology_score"],
        "coalition": CANDIDATES_2026["Paloma Valencia"]["coalition"],
        "endorses": "Abelardo de la Espriella",
    },
    "Sergio Fajardo": {
        **FIRST_ROUND_RESULTS_2026["Sergio Fajardo"],
        "ideology_score": CANDIDATES_2026["Sergio Fajardo"]["ideology_score"],
        "coalition": CANDIDATES_2026["Sergio Fajardo"]["coalition"],
        "endorses": None,  # no official endorsement given
    },
}

# ── Shared helper ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        joined = "\n".join(inner)
        if joined.lstrip().startswith("json"):
            joined = joined.lstrip()[4:]
        return json.loads(joined.strip())
    return json.loads(text)


# ── Agent 1: Updated Voter Bloc Preferences ───────────────────────────────────

def _run_bloc_second_round(bloc_name: str, bloc_data: dict) -> dict:
    """
    Model one congressional bloc's second-round alignment given:
    - Real first-round vote percentages (not just congressional seats)
    - Valencia's official endorsement of Abelardo
    - Fajardo voters as a contested, ideologically centrist pool
    """
    finalist_summary = {
        name: {
            "ideology_score": FINALISTS[name]["ideology_score"],
            "coalition": FINALISTS[name]["coalition"],
            "first_round_pct": FINALISTS[name]["percentage"],
        }
        for name in FINALISTS
    }

    user_message = f"""Model this congressional bloc's voting behavior in Colombia's June 21, 2026 presidential runoff.

SECOND ROUND: {SECOND_ROUND_MATCHUP}

THIS BLOC:
  Name: {bloc_name}
  Senate seats: {bloc_data.get('seats_senate', 0)}
  Ideology score: {bloc_data.get('ideology_score', 0)} (-1 far left, +1 far right)
  Pre-first-round presidential candidate: {bloc_data.get('presidential_candidate', 'TBD')}

FIRST ROUND RESULTS (May 31, 2026):
  Abelardo de la Espriella: 43.74% — advances to second round
  Iván Cepeda:              40.90% — advances to second round
  Paloma Valencia:           6.90% — eliminated, officially endorsed Abelardo
  Sergio Fajardo:            4.26% — eliminated, no official endorsement

FINALIST POSITIONS:
{json.dumps(finalist_summary, ensure_ascii=False, indent=2)}

KEY CONTEXT:
- Valencia (ideology +0.75) has explicitly endorsed Abelardo (ideology +0.55)
- Fajardo (ideology +0.05) is a centrist — his voters are ideologically between the two finalists
- The Liberal party (ideology +0.2) has centrist leanings; their leadership endorsed Valencia in round 1
- Abstention is a real variable — some eliminated candidate voters may not vote in round 2

Return ONLY a JSON object with this exact structure:
{{
  "bloc": "{bloc_name}",
  "seats": {bloc_data.get('seats_senate', 0)},
  "ideology_score": {bloc_data.get('ideology_score', 0)},
  "second_round_preference": "<Abelardo de la Espriella | Iván Cepeda | split | abstain>",
  "preference_probability": <float 0 to 1>,
  "estimated_bloc_split": {{
    "Abelardo de la Espriella": <float 0 to 1>,
    "Iván Cepeda": <float 0 to 1>,
    "abstain": <float 0 to 1>
  }},
  "reasoning": "<2-3 sentences grounded in ideology distance and coalition dynamics>",
  "key_driver": "<the single most important factor for this bloc's alignment>"
}}

No preamble. Only valid JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=VOTER_BLOC_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return _parse_json(response.content[0].text)


def run_updated_voter_blocs() -> list:
    print("\n" + "─" * 70)
    print("AGENT 1: Updated Voter Bloc Preferences (post first-round)")
    print("─" * 70)

    results = []
    for bloc_name, bloc_data in CONGRESSIONAL_RESULTS_2026.items():
        seats = bloc_data["seats_senate"]
        score = bloc_data["ideology_score"]
        print(f"\n  Analyzing: {bloc_name}  ({seats} seats · ideology {score:+.2f})")

        result = _run_bloc_second_round(bloc_name, bloc_data)
        results.append(result)

        pref = result.get("second_round_preference", "?")
        prob = result.get("preference_probability", 0.0)
        split = result.get("estimated_bloc_split", {})
        driver = result.get("key_driver", "")

        print(f"    Preference → {pref}  ({prob:.0%})")
        abelardo_share = split.get("Abelardo de la Espriella", 0)
        cepeda_share = split.get("Iván Cepeda", 0)
        abstain_share = split.get("abstain", 0)
        print(f"    Estimated split: Abelardo {abelardo_share:.0%} · Cepeda {cepeda_share:.0%} · Abstain {abstain_share:.0%}")
        print(f"    Key driver: {driver}")

    return results


# ── Agent 2: Median Voter Recalculation (vote-share weights) ─────────────────

def run_median_voter_second_round(bloc_results: list) -> dict:
    """
    Recalculate the median voter using actual first-round vote percentages
    as weights instead of congressional seat counts.
    """
    print("\n" + "─" * 70)
    print("AGENT 2: Median Voter Recalculation — vote-share weighted")
    print("─" * 70)

    # Build a combined voter universe: finalists + eliminated candidates
    # Each entry has an ideology score and a weight (first-round vote %)
    voter_universe = {
        "Abelardo de la Espriella voters": {
            "ideology_score": CANDIDATES_2026["Abelardo de la Espriella"]["ideology_score"],
            "vote_share": FIRST_ROUND_RESULTS_2026["Abelardo de la Espriella"]["percentage"],
            "second_round_status": "finalist",
        },
        "Iván Cepeda voters": {
            "ideology_score": CANDIDATES_2026["Iván Cepeda"]["ideology_score"],
            "vote_share": FIRST_ROUND_RESULTS_2026["Iván Cepeda"]["percentage"],
            "second_round_status": "finalist",
        },
        "Paloma Valencia voters": {
            "ideology_score": CANDIDATES_2026["Paloma Valencia"]["ideology_score"],
            "vote_share": FIRST_ROUND_RESULTS_2026["Paloma Valencia"]["percentage"],
            "second_round_status": "eliminated — leader endorsed Abelardo",
        },
        "Sergio Fajardo voters": {
            "ideology_score": CANDIDATES_2026["Sergio Fajardo"]["ideology_score"],
            "vote_share": FIRST_ROUND_RESULTS_2026["Sergio Fajardo"]["percentage"],
            "second_round_status": "eliminated — no official endorsement, centrist",
        },
    }

    finalist_positions = {
        name: {
            "ideology_score": FINALISTS[name]["ideology_score"],
            "coalition": FINALISTS[name]["coalition"],
        }
        for name in FINALISTS
    }

    user_message = f"""Recalculate the Median Voter position for Colombia's June 21, 2026 presidential runoff.

SECOND ROUND: {SECOND_ROUND_MATCHUP}
DATE: {SECOND_ROUND_DATE}

VOTER UNIVERSE (weighted by actual May 31st first-round vote shares):
{json.dumps(voter_universe, ensure_ascii=False, indent=2)}

FINALIST IDEOLOGY POSITIONS:
{json.dumps(finalist_positions, ensure_ascii=False, indent=2)}

BLOC SECOND-ROUND ALIGNMENT (from voter bloc analysis):
{json.dumps([{{
    "bloc": r["bloc"],
    "seats": r["seats"],
    "ideology_score": r["ideology_score"],
    "estimated_split": r.get("estimated_bloc_split", {{}}),
}} for r in bloc_results], ensure_ascii=False, indent=2)}

Instructions:
1. Calculate the vote-share-weighted median ideology position across all voter groups
2. Identify which finalist is closer to that median
3. Estimate how Fajardo's 4.26% splits between finalists and abstention — this is the decisive variable
4. Provide a probability estimate for each finalist

Return ONLY a JSON object with this exact structure:
{{
  "median_voter_position": <float -1 to 1>,
  "median_voter_description": "<who is this voter in Colombian political terms>",
  "closer_to_median": "<Abelardo de la Espriella | Iván Cepeda>",
  "fajardo_voter_split_estimate": {{
    "Abelardo de la Espriella": <float 0 to 1>,
    "Iván Cepeda": <float 0 to 1>,
    "abstain": <float 0 to 1>,
    "reasoning": "<why Fajardo voters split this way>"
  }},
  "second_round_prediction": {{
    "predicted_winner": "<candidate name>",
    "win_probability": <float 0.5 to 1.0>,
    "margin_estimate": "<narrow | moderate | comfortable>",
    "confidence": "<low | medium | high>"
  }},
  "abelardo_path_to_victory": "<2-3 sentences>",
  "cepeda_path_to_victory": "<2-3 sentences>",
  "decisive_variable": "<the single factor that most determines the outcome>"
}}

No preamble. Only valid JSON."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=MEDIAN_VOTER_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    result = _parse_json(response.content[0].text)

    pos = result.get("median_voter_position", 0.0)
    desc = result.get("median_voter_description", "")
    closer = result.get("closer_to_median", "?")
    print(f"\n📍 Median Voter Position: {pos:+.3f}")
    print(f"   {desc}")
    print(f"   Closer to median: {closer}")

    fajardo_split = result.get("fajardo_voter_split_estimate", {})
    print(f"\n🔀 Fajardo Voter Split (4.26% of total vote — decisive pool):")
    for key in ("Abelardo de la Espriella", "Iván Cepeda", "abstain"):
        pct = fajardo_split.get(key, 0.0)
        print(f"   {key:<35} {pct:.0%}")
    print(f"   Reasoning: {fajardo_split.get('reasoning', '')}")

    pred = result.get("second_round_prediction", {})
    winner = pred.get("predicted_winner", "?")
    win_prob = pred.get("win_probability", 0.0)
    margin = pred.get("margin_estimate", "?")
    confidence = pred.get("confidence", "?")
    print(f"\n🎯 Second Round Prediction:")
    print(f"   Winner   : {winner}  ({win_prob:.0%})")
    print(f"   Margin   : {margin}")
    print(f"   Confidence: {confidence}")

    print(f"\n   Abelardo's path: {result.get('abelardo_path_to_victory', '')}")
    print(f"\n   Cepeda's path:   {result.get('cepeda_path_to_victory', '')}")
    print(f"\n⚡ Decisive variable: {result.get('decisive_variable', '')}")

    return result


# ── Orchestrator: Final Second-Round Synthesis ───────────────────────────────

def run_second_round_orchestrator() -> dict:
    all_results: dict = {}

    # Agent 1: Updated bloc preferences
    bloc_results = run_updated_voter_blocs()
    all_results["updated_voter_blocs"] = bloc_results

    # Agent 2: Median voter with vote-share weights
    median_result = run_median_voter_second_round(bloc_results)
    all_results["median_voter_second_round"] = median_result

    # Final synthesis
    print("\n" + "─" * 70)
    print("ORCHESTRATOR: Second Round Synthesis")
    print("─" * 70)

    bloc_summary = "\n".join(
        f"  - {r['bloc']} ({r['seats']} seats): "
        f"split Abelardo {r.get('estimated_bloc_split', {}).get('Abelardo de la Espriella', 0):.0%} / "
        f"Cepeda {r.get('estimated_bloc_split', {}).get('Iván Cepeda', 0):.0%} / "
        f"abstain {r.get('estimated_bloc_split', {}).get('abstain', 0):.0%}"
        for r in bloc_results
    )

    pred = median_result.get("second_round_prediction", {})
    fajardo = median_result.get("fajardo_voter_split_estimate", {})

    synthesis_prompt = f"""Synthesize a neutral second-round analysis for Colombia's June 21, 2026 presidential runoff:
{SECOND_ROUND_MATCHUP}

FIRST ROUND RESULTS (May 31, 2026):
  Abelardo de la Espriella: 43.74%  (advances)
  Iván Cepeda:              40.90%  (advances)
  Paloma Valencia:           6.90%  (eliminated — endorsed Abelardo)
  Sergio Fajardo:            4.26%  (eliminated — no endorsement)

GAP TO CLOSE: Cepeda trails by 2.84 percentage points.
VALENCIA ENDORSEMENT: Adds institutional right-wing support to Abelardo.
FAJARDO POOL (4.26%): Centrist voters — estimated split:
  → Abelardo {fajardo.get("Abelardo de la Espriella", 0):.0%} / Cepeda {fajardo.get("Iván Cepeda", 0):.0%} / abstain {fajardo.get("abstain", 0):.0%}

CONGRESSIONAL BLOC SECOND-ROUND SPLITS:
{bloc_summary}

MEDIAN VOTER:
  Position: {median_result.get("median_voter_position", 0):+.3f}
  Closer to: {median_result.get("closer_to_median", "?")}

PREDICTION: {pred.get("predicted_winner", "?")} — {pred.get("win_probability", 0):.0%} probability — {pred.get("margin_estimate", "?")} margin

Abelardo's path: {median_result.get("abelardo_path_to_victory", "")}
Cepeda's path:   {median_result.get("cepeda_path_to_victory", "")}
Decisive variable: {median_result.get("decisive_variable", "")}

Write 3 neutral paragraphs:
1. The structural dynamics: what the first-round gap, Valencia endorsement, and Fajardo pool mean for the runoff
2. Abelardo's path to victory vs Cepeda's path — what each candidate needs
3. The decisive variable: abstention among Fajardo and Valencia voters, and what June 21st likely produces"""

    synthesis_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=ORCHESTRATOR_PROMPT,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesis_text = synthesis_response.content[0].text.strip()
    print(f"\n{synthesis_text}")
    all_results["synthesis"] = synthesis_text

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "second_round_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SEGUNDA VUELTA — 21 DE JUNIO 2026")
    print("Abelardo de la Espriella vs Iván Cepeda")
    print("Based on May 31st first round results")
    print("=" * 70)

    run_second_round_orchestrator()

    print("\n" + "=" * 70)
    print("All results saved to output/second_round_results.json")
    print("=" * 70)
