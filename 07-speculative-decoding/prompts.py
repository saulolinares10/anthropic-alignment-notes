"""
XML prompt templates for the speculative decoding acceptance rate profiler.
Three task types mirroring M&A Diligence sub-agents.
Cuesta structure: <role><context><task><format>
"""

import random

# ──────────────────────────────────────────────────────────────
# TYPE A — Structured JSON extraction (financial parsing sub-agent)
# Highest expected α: constrained token space → draft model accurate
# ──────────────────────────────────────────────────────────────

COMPANIES = [
    "TechCorp Industries", "MedDevice Solutions", "RetailChain Holdings",
    "SoftwareCo Inc", "ManufacturingPlus LLC", "HealthNet Services",
    "DataStream Analytics", "CloudBase Systems", "NanoTech Ventures",
    "LogiFlow Corp", "FinServe Global", "AutoParts Direct",
    "BioPharma Labs", "EnergyGrid Solutions", "ConsumerGoods Co",
    "AgriTech Partners", "CyberSec Defense", "MediaStream Inc",
    "RealEstate Capital", "TransportLink Ltd",
]

SECTORS = [
    "SaaS", "Medical Devices", "Retail", "Enterprise Software",
    "Industrial Manufacturing", "Healthcare Services", "Data Analytics",
    "Cloud Infrastructure", "Deep Tech", "Logistics",
]


def make_json_prompt(seed_offset: int = 0) -> str:
    rng = random.Random(42 + seed_offset)
    company = rng.choice(COMPANIES)
    sector = rng.choice(SECTORS)
    revenue = rng.randint(15, 500)
    ebitda = round(revenue * rng.uniform(0.08, 0.28), 1)
    growth = round(rng.uniform(-0.05, 0.35) * 100, 1)
    churn = round(rng.uniform(0.04, 0.22) * 100, 1)
    nwc = round(rng.uniform(-0.12, 0.18) * revenue, 1)

    return f"""<role>
You are a financial data extraction engine for an M&A diligence platform.
Extract structured financial metrics from unstructured deal descriptions.
</role>
<context>
Target company: {company} ({sector})
Deal stage: Initial screening
Data source: Management presentation excerpt
</context>
<task>
Extract the following financial metrics from the passage below and return them
as a valid JSON object with exactly these keys:
- company_name (string)
- sector (string)
- revenue_mm (number, USD millions)
- ebitda_mm (number, USD millions)
- ebitda_margin_pct (number, percent)
- revenue_growth_pct (number, percent, YoY)
- churn_rate_pct (number, percent, annualized)
- nwc_mm (number, USD millions, positive = asset)

Passage:
"{company} reported ${revenue}M in revenue for the trailing twelve months,
with EBITDA of ${ebitda}M. Year-over-year revenue growth was {growth}%.
Customer churn stands at {churn}% on an annualized basis. Net working capital
is ${abs(nwc)}M {'(asset)' if nwc >= 0 else '(liability)'}."
</task>
<format>
Return ONLY a valid JSON object. No markdown fences. No explanation.
All numeric values must be numbers (not strings). Null for missing fields.
</format>"""


# ──────────────────────────────────────────────────────────────
# TYPE B — Templated markdown table (benchmarking sub-agent)
# Medium expected α: structured but with variable content
# ──────────────────────────────────────────────────────────────

METRICS = [
    ("Revenue Growth (YoY)", "%"),
    ("EBITDA Margin", "%"),
    ("Gross Margin", "%"),
    ("Customer Churn", "% annualized"),
    ("Net Revenue Retention", "%"),
    ("CAC Payback", "months"),
    ("Rule of 40 Score", "points"),
    ("Debt / EBITDA", "x"),
]

PEERS = [
    ["AlphaCo", "BetaCorp", "GammaTech"],
    ["NovaSoft", "OmegaData", "PrimeSys"],
    ["QuantumIO", "RapidScale", "SkyBridge"],
    ["TerraCloud", "UnityAI", "VectorOne"],
    ["WaveTech", "XcelSoft", "YieldMax"],
]


def make_table_prompt(seed_offset: int = 0) -> str:
    rng = random.Random(42 + seed_offset)
    company = rng.choice(COMPANIES)
    peers = rng.choice(PEERS)
    selected_metrics = rng.sample(METRICS, 5)
    metric_lines = "\n".join(
        f"- {m[0]} ({m[1]}): target={round(rng.uniform(5,40),1)}, "
        f"peer avg={round(rng.uniform(4,38),1)}"
        for m in selected_metrics
    )

    return f"""<role>
You are a competitive benchmarking analyst preparing materials for an
investment committee presentation. You produce clean, publication-ready
markdown tables.
</role>
<context>
Target company: {company}
Peer group: {', '.join(peers)}
Analysis purpose: Pre-LOI competitive positioning memo
Audience: Investment Committee
</context>
<task>
Produce a markdown comparison table for {company} vs. the peer group using
the metrics below. Include a "vs. Peer Avg" delta column (+ or - with sign).
Add a brief 1-sentence **Positioning** note below the table summarizing
{company}'s relative standing.

Metrics:
{metric_lines}
</task>
<format>
Return a valid GitHub-flavored markdown table followed by the Positioning line.
Table header: | Metric | Unit | {company} | Peer Avg | vs. Peer Avg |
Use | --- | alignment markers. No extra prose beyond the table and Positioning.
</format>"""


# ──────────────────────────────────────────────────────────────
# TYPE C — Open-ended narrative paragraph (IC memo drafting sub-agent)
# Lowest expected α: unconstrained token space → draft drifts from target
# ──────────────────────────────────────────────────────────────

RISKS = [
    "customer concentration (top 3 customers = 62% of ARR)",
    "founder dependency (CEO sole technical decision-maker, no VP Eng)",
    "EBITDA adjustments (40% add-back rate, primarily one-time items)",
    "NWC normalization gap ($8M liability vs. $2M target in LOI)",
    "competitive pressure from well-funded entrants in core segment",
    "data security exposure (SOC 2 Type II pending, no GDPR certification)",
    "churn acceleration risk (2 anchor clients up for renewal in Q3)",
    "geographic concentration (78% US revenue, limited international presence)",
]

DEAL_TYPES = [
    "growth equity investment", "leveraged buyout", "strategic acquisition",
    "minority recapitalization", "platform add-on",
]


def make_narrative_prompt(seed_offset: int = 0) -> str:
    rng = random.Random(42 + seed_offset)
    company = rng.choice(COMPANIES)
    sector = rng.choice(SECTORS)
    deal_type = rng.choice(DEAL_TYPES)
    risk1, risk2 = rng.sample(RISKS, 2)
    revenue = rng.randint(20, 300)
    multiple = round(rng.uniform(4.5, 12.0), 1)

    return f"""<role>
You are a senior associate at a private equity firm drafting an Investment
Committee memo. Your writing is precise, risk-aware, and investment-grade —
you surface concerns explicitly rather than burying them in qualifications.
</role>
<context>
Deal: {deal_type} in {company} ({sector})
Revenue: ${revenue}M TTM
Entry multiple: {multiple}x EV/Revenue
Key diligence findings: {risk1}; {risk2}
</context>
<task>
Write a single Investment Considerations paragraph (120–160 words) for the
IC memo covering {company}. The paragraph must:
1. Open with the investment thesis in one sentence
2. Acknowledge both diligence risks explicitly by name
3. State what additional diligence would be required before proceeding
4. End with a clear recommendation (proceed / proceed with conditions / pass)
</task>
<format>
Plain prose paragraph. No bullet points. No headers. 120–160 words.
Investment-grade register: precise, factual, risk-forward.
</format>"""


# ──────────────────────────────────────────────────────────────
# Generator: produce all 60 prompts
# ──────────────────────────────────────────────────────────────

def get_all_prompts(n_per_type: int = 20) -> dict:
    """Return dict with keys 'json', 'table', 'narrative', each a list of prompts."""
    return {
        "json":      [make_json_prompt(i)      for i in range(n_per_type)],
        "table":     [make_table_prompt(i)     for i in range(n_per_type)],
        "narrative": [make_narrative_prompt(i) for i in range(n_per_type)],
    }


TASK_LABELS = {
    "json":      "Type A: JSON Extraction (Financial Parsing)",
    "table":     "Type B: Markdown Table (Benchmarking)",
    "narrative": "Type C: Narrative Paragraph (IC Memo Drafting)",
}
