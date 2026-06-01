"""
Colombia 2026 segunda vuelta data.
Sources: Registraduría Nacional (May 31 results), Invamer/CNN, AtlasIntel/Semana, Guarumo/Ecoanalítica.
Real numbers only. No congressional seat proxies.
"""

FIRST_ROUND_RESULTS = {
    "Abelardo de la Espriella": {
        "votes": 10_361_413,
        "pct": 43.74,
        "ideology": 0.75,
        "coalition": "Defensores de la Patria",
        "second_round": True,
    },
    "Iván Cepeda": {
        "votes": 9_688_245,
        "pct": 40.90,
        "ideology": -0.80,
        "coalition": "Pacto Histórico",
        "second_round": True,
    },
    "Paloma Valencia": {
        "votes": 1_625_563,
        "pct": 6.92,
        "ideology": 0.82,
        "coalition": "Centro Democrático",
        "second_round": False,
        "endorsed": "Abelardo de la Espriella",
        "endorsement_efficiency_estimate": 0.70,
    },
    "Sergio Fajardo": {
        "votes": 1_000_974,
        "pct": 4.26,
        "ideology": 0.05,
        "coalition": "Centro Esperanza",
        "second_round": False,
        "endorsed": None,
        "notes": "Centro pragmático. Históricamente anti-uribismo pero distante del Pacto Histórico.",
    },
    "Claudia López": {
        "votes": 224_000,
        "pct": 0.95,
        "ideology": -0.10,
        "coalition": "Imparables",
        "second_round": False,
        "endorsed": None,
    },
    "Otros": {
        "pct": 3.23,
        "ideology": 0.20,
        "second_round": False,
    },
}

SECOND_ROUND_POLLS = [
    {
        "firm": "Invamer",
        "date": "Mayo 2026",
        "abelardo": 45.3,
        "cepeda": 52.4,
        "blanco_nulo": 2.3,
        "predicted_winner": "Cepeda",
        "source": "CNN/Invamer",
    },
    {
        "firm": "AtlasIntel / Semana",
        "date": "Mayo 2026",
        "abelardo": 50.0,
        "cepeda": 41.3,
        "blanco_nulo": 8.8,
        "predicted_winner": "Abelardo",
        "source": "Revista Semana",
    },
    {
        "firm": "Guarumo / Ecoanalítica",
        "date": "Mayo 2026",
        "abelardo": 44.8,
        "cepeda": 39.9,
        "blanco_nulo": 15.3,
        "predicted_winner": "Abelardo",
        "source": "El Tiempo",
        "notes": "Escenario Cepeda-Abelardo directo",
    },
]

HISTORICAL_CONTEXT = {
    "2022_first_round": {
        "Petro": 40.32,
        "Hernandez": 28.15,
        "Gutierrez": 23.92,
        "Fajardo": 4.20,
    },
    "2022_second_round": {
        "Petro": 50.44,
        "Hernandez": 47.31,
        "winner": "Petro",
        "margin": 3.13,
        "lesson": (
            "Left coalition mobilized better in second round. "
            "Fajardo voters split ~60% Petro, 40% Hernandez. "
            "Abstention shifted toward Petro."
        ),
    },
    "abstention_rate_historically": 0.45,
    "second_round_mobilization_pattern": (
        "Left historically mobilizes better in second rounds in Colombia. "
        "2022 was decisive example."
    ),
}

SWING_VARIABLES = {
    "paloma_transfer": {
        "votes_available": 1_625_563,
        "pct_available": 6.92,
        "formal_endorsement": "Abelardo",
        "estimated_effective_transfer": {
            "optimistic_abelardo": 0.80,
            "realistic_abelardo": 0.65,
            "pessimistic_abelardo": 0.50,
        },
        "reasoning": (
            "Formal endorsement helps but CD voters in urban areas may not follow. "
            "Some CD voters ideologically closer to center."
        ),
    },
    "fajardo_transfer": {
        "votes_available": 1_000_974,
        "pct_available": 4.26,
        "formal_endorsement": None,
        "estimated_split": {
            "abelardo": 0.35,
            "cepeda": 0.30,
            "abstention": 0.35,
        },
        "reasoning": (
            "Fajardo voters are centrists. Historically anti-uribismo "
            "(won't fully go to Abelardo) but also skeptical of Pacto Histórico. "
            "High abstention expected."
        ),
    },
    "abstention_differential": {
        "base_abstention": 0.45,
        "scenarios": {
            "left_mobilizes_better": "Cepeda +3 to +5 points vs current polls",
            "right_mobilizes_better": "Abelardo +2 to +4 points vs current polls",
            "equal_mobilization": "Polls hold as reference",
        },
        "historical_note": (
            "In 2022 second round, left mobilization was decisive. "
            "Petro gained ~10 points from first to second round."
        ),
    },
    "cepeda_rejection_factor": {
        "note": "Cepeda is seen as Petro's candidate. Anti-Petrismo is a real mobilization force for Abelardo.",
        "estimated_impact": "Adds 3-5% to Abelardo's base beyond pure ideology",
    },
}

CANDIDATE_IDEOLOGY = {
    "Abelardo": 0.75,
    "Cepeda": -0.80,
    "Paloma": 0.82,
    "Fajardo": 0.05,
    "Lopez": -0.10,
    "median_voter_estimate": 0.15,
    "median_voter_description": (
        "Center-right, urban, skeptical of Petro but not fully Uribista. "
        "Values stability and anti-corruption."
    ),
}
