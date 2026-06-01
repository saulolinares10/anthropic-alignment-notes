"""
Colombia 2026 presidential election data.
Sources: Registraduría Nacional (congressional results, March 8 2026);
El Tiempo, Razón Pública, Bloomberg Línea, Infobae Colombia (candidate proposals, May 2026).
"""

CONGRESSIONAL_RESULTS_2026 = {
    "Pacto Histórico": {
        "seats_senate": 25,
        "votes": 4_413_636,
        "ideology_score": -0.8,   # -1 far left, +1 far right
        "presidential_candidate": "Iván Cepeda",
    },
    "Centro Democrático": {
        "seats_senate": 17,
        "votes": 3_035_715,
        "ideology_score": 0.85,
        "presidential_candidate": "Paloma Valencia",
    },
    "Partido Liberal": {
        "seats_senate": 11,
        "votes": 2_275_182,
        "ideology_score": 0.2,
        "presidential_candidate": "Paloma Valencia (endorsed)",
    },
    "Alianza por Colombia": {
        "seats_senate": 9,
        "votes": 1_904_154,
        "ideology_score": 0.5,
        "presidential_candidate": "TBD",
    },
    "Partido Conservador": {
        "seats_senate": 8,
        "votes": 1_863_663,
        "ideology_score": 0.7,
        "presidential_candidate": "TBD",
    },
}

CANDIDATES_2026 = {
    "Paloma Valencia": {
        "coalition": "Centro Democrático + Liberal",
        "ideology_score": 0.75,
        "key_proposals": {
            "economy": (
                "Reducir impuesto de renta empresarial, eliminar impuesto al patrimonio, "
                "zonas francas regionales, TLCs con Asia. Meta: crecimiento 5% anual."
            ),
            "security": (
                "Seguridad Total — 30,000 nuevos militares y 30,000 policías. "
                "Plan 30-30. Recuperación territorial."
            ),
            "health": "Resolver millones de citas represadas en primeros meses. Fortalecer cobertura.",
            "education": "Bonos escolares, becas, formación tecnológica en IA.",
            "energy": (
                "Apoya fracking bajo regulación ambiental. "
                "Fortalecer producción energética para financiar programas sociales."
            ),
        },
    },
    "Iván Cepeda": {
        "coalition": "Pacto Histórico",
        "ideology_score": -0.75,
        "key_proposals": {
            "economy": (
                "Reforma tributaria progresiva. Continuidad de políticas del gobierno Petro. "
                "Estado como motor de desarrollo."
            ),
            "security": (
                "Paz Total — negociación con grupos armados. "
                "Enfoque en causas estructurales de la violencia."
            ),
            "health": "Reforma a la salud. Sistema público fortalecido.",
            "education": "Educación pública gratuita. Universidad para todos.",
            "energy": "Transición energética acelerada. No al fracking.",
        },
    },
    "Claudia López": {
        "coalition": "Consulta de las Soluciones",
        "ideology_score": -0.1,
        "key_proposals": {
            "economy": (
                "Presupuestos participativos regionales. Descentralización fiscal. "
                "Lucha anticorrupción como eje."
            ),
            "security": (
                "Redefine violencia como crimen organizado. "
                "Enfoque en ciudades y comunidades."
            ),
            "health": "Recuperar el sistema de salud. Reforma moderada.",
            "education": "Educación de calidad con énfasis en regiones.",
            "energy": "Transición energética gradual con responsabilidad fiscal.",
        },
    },
    "Abelardo de la Espriella": {
        "coalition": "Independiente centro-derecha",
        "ideology_score": 0.55,
        "key_proposals": {
            "economy": "Crecimiento empresarial, reducción de trámites, apoyo a pymes.",
            "security": "Mano dura contra el crimen. Recuperación del orden.",
            "health": "Sistema mixto público-privado.",
            "education": "Educación técnica y vocacional.",
            "energy": "Balance entre producción y transición.",
        },
    },
}

POLICY_DIMENSIONS = [
    "economy",
    "security",
    "health",
    "education",
    "energy",
]
