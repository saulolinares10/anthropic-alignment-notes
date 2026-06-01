"""
Colombia 2026 · Segunda Vuelta Simulator
=========================================
Abelardo de la Espriella vs Iván Cepeda · June 21, 2026

Three-agent pipeline:
  1. Survey Analyst    — reconciles divergent polls (Invamer, AtlasIntel, Guarumo)
  2. Swing Analyst     — models Paloma transfer, Fajardo split, abstention differential
  3. Median Voter Agent — applies MVT and identifies Colombian deviations from theory

Built June 1st, 2026. Uses only verified first-round vote data and published polls.
No congressional seat proxies.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python simulator.py
"""

import json
from pathlib import Path

import anthropic

from data import (
    CANDIDATE_IDEOLOGY,
    FIRST_ROUND_RESULTS,
    HISTORICAL_CONTEXT,
    SECOND_ROUND_POLLS,
    SWING_VARIABLES,
)
from prompts import (
    MEDIAN_VOTER_PROMPT,
    ORCHESTRATOR_PROMPT,
    SURVEY_ANALYST_PROMPT,
    SWING_ANALYST_PROMPT,
)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text):
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Return raw text if all else fails
        return {"raw_response": text}


def call_agent(system_prompt, user_message, agent_name):
    print(f"\n{'─' * 70}")
    print(f"AGENT: {agent_name}")
    print("─" * 70)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── Agent 1: Survey Analyst ───────────────────────────────────────────────────

def run_survey_analyst() -> dict:
    """
    Reconcile three divergent polls, produce a weighted consensus,
    and identify what polling cannot capture.
    """
    user_message = f"""Analyze three divergent second-round poll estimates for Colombia's June 21, 2026 presidential runoff.

POLLS:
{json.dumps(SECOND_ROUND_POLLS, ensure_ascii=False, indent=2)}

FIRST ROUND RESULT (actual, May 31):
  Abelardo de la Espriella: 43.74%
  Iván Cepeda:              40.90%
  Gap: +2.84 points for Abelardo

Note: These polls produce opposing winners (Invamer: Cepeda; AtlasIntel + Guarumo: Abelardo).

Return ONLY a JSON object with this exact structure:
{{
  "poll_divergence_reason": "<2-3 sentences explaining why these polls produce different winners>",
  "weighted_consensus": {{
    "abelardo": <float>,
    "cepeda": <float>,
    "margin": <float — positive means Abelardo leads>,
    "uncertainty_band": "<e.g. ±3.5 points>"
  }},
  "what_polls_cannot_tell_us": ["<limitation 1>", "<limitation 2>", "<limitation 3>"],
  "most_credible_scenario": "<which poll or combination is most credible and why>"
}}

No preamble. Only valid JSON."""

    raw = call_agent(SURVEY_ANALYST_PROMPT, user_message, "Survey Analyst")
    result = _parse_json(raw)

    # Print poll comparison table
    print(f"\n  {'Firm':<28} {'Abelardo':>10} {'Cepeda':>10} {'B/N':>8}  Winner")
    print("  " + "─" * 64)
    for p in SECOND_ROUND_POLLS:
        print(
            f"  {p['firm'][:27]:<28}"
            f"  {p['abelardo']:.1f}%{'':<5}"
            f"  {p['cepeda']:.1f}%{'':<5}"
            f"  {p['blanco_nulo']:.1f}%{'':<3}"
            f"  {p['predicted_winner']}"
        )

    wc = result.get("weighted_consensus", {})
    margin = wc.get("margin", 0)
    leader = "Abelardo" if margin >= 0 else "Cepeda"
    print(
        f"\n  Weighted consensus: Abelardo {wc.get('abelardo', 0):.1f}% — "
        f"Cepeda {wc.get('cepeda', 0):.1f}%"
    )
    print(f"  Margin: {leader} +{abs(margin):.1f}%  (uncertainty: {wc.get('uncertainty_band', '?')})")
    print(f"\n  Divergence reason: {result.get('poll_divergence_reason', '')}")

    print("\n  What polls cannot tell us:")
    for lim in result.get("what_polls_cannot_tell_us", []):
        print(f"    · {lim}")

    print(f"\n  Most credible scenario: {result.get('most_credible_scenario', '')}")

    return result


# ── Agent 2: Swing Variable Analyst ──────────────────────────────────────────

def run_swing_analyst() -> dict:
    """
    Model three vote-transfer scenarios using Paloma endorsement efficiency,
    Fajardo centrist split, and 2022 abstention patterns as anchors.
    """
    user_message = f"""Model vote transfer scenarios for Colombia's June 21, 2026 presidential runoff.

BASELINE (May 31 first round):
  Abelardo: 43.74%  (10,361,413 votes)
  Cepeda:   40.90%  ( 9,688,245 votes)
  Gap: +2.84 points for Abelardo

SWING VARIABLES:
{json.dumps(SWING_VARIABLES, ensure_ascii=False, indent=2)}

HISTORICAL ANCHOR — 2022 second round:
{json.dumps(HISTORICAL_CONTEXT["2022_second_round"], ensure_ascii=False, indent=2)}

IDEOLOGY REFERENCE:
  Abelardo: +0.75 (right)
  Cepeda:   -0.80 (left)
  Paloma:   +0.82 (further right than Abelardo — endorsement is natural)
  Fajardo:  +0.05 (center, closer to Abelardo in position, but historically anti-Uribista)
  Median voter: +0.15 (center-right)

Return ONLY a JSON object with this exact structure:
{{
  "scenario_abelardo_wins": {{
    "paloma_transfer_pct": <fraction of Paloma's 6.92% that goes to Abelardo>,
    "fajardo_to_abelardo_pct": <fraction of Fajardo's 4.26% that goes to Abelardo>,
    "abstention_pattern": "<brief description>",
    "abelardo_final": <estimated final vote share>,
    "cepeda_final": <estimated final vote share>,
    "probability": <float 0 to 1>
  }},
  "scenario_cepeda_wins": {{
    "paloma_transfer_pct": <fraction of Paloma's 6.92% going to Abelardo>,
    "fajardo_to_cepeda_pct": <fraction of Fajardo's 4.26% going to Cepeda>,
    "abstention_pattern": "<brief description>",
    "abelardo_final": <estimated final vote share>,
    "cepeda_final": <estimated final vote share>,
    "probability": <float 0 to 1>
  }},
  "scenario_toss_up": {{
    "description": "<what makes this a true toss-up>",
    "deciding_variable": "<the single variable that tips it>",
    "abelardo_final": <estimated final vote share>,
    "cepeda_final": <estimated final vote share>,
    "probability": <float 0 to 1>
  }},
  "most_likely_scenario": "<scenario_abelardo_wins | scenario_cepeda_wins | scenario_toss_up>",
  "key_insight": "<the single most important insight from this vote-transfer analysis>"
}}

Probabilities must sum to 1.0. No preamble. Only valid JSON."""

    raw = call_agent(SWING_ANALYST_PROMPT, user_message, "Swing Variable Analyst")
    result = _parse_json(raw)

    # Print three scenarios
    for label, key in [
        ("ABELARDO WINS", "scenario_abelardo_wins"),
        ("CEPEDA WINS",   "scenario_cepeda_wins"),
        ("TOSS-UP",       "scenario_toss_up"),
    ]:
        s = result.get(key, {})
        ab = s.get("abelardo_final", 0)
        ce = s.get("cepeda_final", 0)
        prob = s.get("probability", 0)
        print(f"\n  SCENARIO: {label}  (probability: {prob:.0%})")
        print(f"    Abelardo {ab:.1f}%  —  Cepeda {ce:.1f}%")

        if key == "scenario_abelardo_wins":
            print(f"    Paloma → Abelardo: {s.get('paloma_transfer_pct', 0):.0%} of her vote")
            print(f"    Fajardo → Abelardo: {s.get('fajardo_to_abelardo_pct', 0):.0%} of his vote")
            print(f"    Abstention: {s.get('abstention_pattern', '')}")
        elif key == "scenario_cepeda_wins":
            print(f"    Paloma → Abelardo: {s.get('paloma_transfer_pct', 0):.0%} of her vote")
            print(f"    Fajardo → Cepeda: {s.get('fajardo_to_cepeda_pct', 0):.0%} of his vote")
            print(f"    Abstention: {s.get('abstention_pattern', '')}")
        else:
            print(f"    Deciding variable: {s.get('deciding_variable', '')}")
            print(f"    {s.get('description', '')}")

    print(f"\n  Most likely scenario: {result.get('most_likely_scenario', '?')}")
    print(f"  Key insight: {result.get('key_insight', '')}")

    return result


# ── Agent 3: Median Voter Agent ───────────────────────────────────────────────

def run_median_voter_agent() -> dict:
    """
    Apply the Median Voter Theorem using first-round vote shares as weights,
    then identify where Colombian political reality deviates from the theorem.
    """
    user_message = f"""Apply the Median Voter Theorem to Colombia's June 21, 2026 presidential runoff.

CANDIDATE IDEOLOGY POSITIONS:
{json.dumps(CANDIDATE_IDEOLOGY, ensure_ascii=False, indent=2)}

FIRST ROUND VOTE SHARES (use as voter weight approximations):
  Abelardo (+0.75):  43.74%
  Cepeda   (-0.80):  40.90%
  Paloma   (+0.82):   6.92%  → endorsed Abelardo
  Fajardo  (+0.05):   4.26%  → no endorsement
  López    (-0.10):   0.95%  → no endorsement
  Otros    (+0.20):   3.23%

HISTORICAL CONTEXT:
{json.dumps(HISTORICAL_CONTEXT, ensure_ascii=False, indent=2)}

Instructions:
1. Calculate the vote-weighted median ideological position
2. Identify which candidate is closer to that median
3. State the pure MVT prediction
4. Identify where Colombian political reality deviates from MVT assumptions
5. Give an MVT-adjusted prediction that accounts for those deviations

Return ONLY a JSON object with this exact structure:
{{
  "median_voter_position": <float — calculated from vote-weighted distribution>,
  "closest_candidate": "<Abelardo | Cepeda>",
  "mvt_prediction": "<pure MVT prediction and one-sentence reasoning>",
  "colombian_deviations": [
    {{
      "deviation": "<name of the deviation from MVT assumptions>",
      "impact_on_mvt": "<how this undermines or modifies the pure prediction>",
      "favors": "<Abelardo | Cepeda | neutral>"
    }}
  ],
  "mvt_adjusted_prediction": "<adjusted prediction accounting for all identified deviations>",
  "confidence": "<low | medium | high — with one-sentence justification>"
}}

No preamble. Only valid JSON."""

    raw = call_agent(MEDIAN_VOTER_PROMPT, user_message, "Median Voter Agent")
    result = _parse_json(raw)

    pos = result.get("median_voter_position", 0.0)
    closest = result.get("closest_candidate", "?")
    pure = result.get("mvt_prediction", "")
    adjusted = result.get("mvt_adjusted_prediction", "")
    confidence = result.get("confidence", "?")

    print(f"\n  Median voter position: {pos:+.3f}")
    print(f"  Closest candidate:     {closest}")
    print(f"\n  Pure MVT prediction: {pure}")

    print("\n  Colombian deviations from MVT:")
    for d in result.get("colombian_deviations", []):
        print(f"    · {d.get('deviation', '?')}")
        print(f"      Impact: {d.get('impact_on_mvt', '')}")
        print(f"      Favors: {d.get('favors', '?')}")

    print(f"\n  MVT-adjusted prediction: {adjusted}")
    print(f"  Confidence: {confidence}")

    return result


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_orchestrator(survey_result: dict, swing_result: dict, mvt_result: dict) -> str:
    """
    Synthesize all three agent outputs into a 2-paragraph neutral assessment
    with explicit probability estimate and confidence interval.
    """
    print(f"\n{'─' * 70}")
    print("ORCHESTRATOR: Final Synthesis")
    print("─" * 70)

    wc = survey_result.get("weighted_consensus", {})
    sw_likely = swing_result.get("most_likely_scenario", "?")
    sw_insight = swing_result.get("key_insight", "")
    sw_a = swing_result.get("scenario_abelardo_wins", {})
    sw_c = swing_result.get("scenario_cepeda_wins", {})
    sw_t = swing_result.get("scenario_toss_up", {})
    mvt_adj = mvt_result.get("mvt_adjusted_prediction", "")
    mvt_conf = mvt_result.get("confidence", "?")
    deviations = [d.get("deviation", "") for d in mvt_result.get("colombian_deviations", [])]

    user_message = f"""Synthesize three analyses of Colombia's June 21, 2026 presidential runoff.

SURVEY ANALYSIS:
  Weighted consensus: Abelardo {wc.get('abelardo', 0):.1f}% — Cepeda {wc.get('cepeda', 0):.1f}%
  Margin: {wc.get('margin', 0):+.1f} pts  (uncertainty: {wc.get('uncertainty_band', '?')})
  Most credible scenario: {survey_result.get('most_credible_scenario', '?')}
  Key poll limitation: {survey_result.get('what_polls_cannot_tell_us', ['?'])[0]}

SWING VARIABLE ANALYSIS:
  Most likely scenario: {sw_likely}
  Abelardo wins ({sw_a.get('probability', 0):.0%}): Abelardo {sw_a.get('abelardo_final', 0):.1f}% — Cepeda {sw_c.get('cepeda_final', 0):.1f}%
  Cepeda wins  ({sw_c.get('probability', 0):.0%}): Abelardo {sw_c.get('abelardo_final', 0):.1f}% — Cepeda {sw_c.get('cepeda_final', 0):.1f}%
  Toss-up      ({sw_t.get('probability', 0):.0%}): deciding variable — {sw_t.get('deciding_variable', '?')}
  Key insight: {sw_insight}

MEDIAN VOTER ANALYSIS:
  MVT-adjusted prediction: {mvt_adj}
  Confidence: {mvt_conf}
  Colombian deviations identified: {', '.join(deviations)}

Write exactly 2 paragraphs:
1. What the three analyses agree on, and the single most decisive variable for June 21st.
2. What remains genuinely uncertain, and a probability estimate with explicit confidence interval.

Be direct. Acknowledge uncertainty. No hedging beyond what the data supports."""

    raw = call_agent(ORCHESTRATOR_PROMPT, user_message, "Orchestrator — Final Synthesis")
    print(f"\n{raw}")
    return raw


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("COLOMBIA · SEGUNDA VUELTA · 21 DE JUNIO 2026")
    print("Abelardo de la Espriella vs Iván Cepeda")
    print("Análisis multi-agente basado en encuestas y datos reales")
    print("=" * 70)

    survey = run_survey_analyst()
    swing  = run_swing_analyst()
    mvt    = run_median_voter_agent()
    synthesis = run_orchestrator(survey, swing, mvt)

    # Save all results
    all_results = {
        "survey_analysis": survey,
        "swing_analysis": swing,
        "mvt_analysis": mvt,
        "synthesis": synthesis,
    }

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "segunda_vuelta_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("All results saved to output/segunda_vuelta_analysis.json")
    print("=" * 70)
