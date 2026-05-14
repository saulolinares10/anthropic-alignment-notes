import json, os

def md(source): return {"cell_type":"markdown","metadata":{},"source":source}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""# Transformer Architecture — Fundamentals

**Paper:** Vaswani et al., *Attention Is All You Need* (2017)

This notebook covers the mechanics of transformer models from the ground up: how text becomes tokens, how tokens become vectors with semantic geometry, how attention routes information across a sequence, and why context windows have hard limits. Most sections use only NumPy and matplotlib — the goal is to see the math directly, not to call a framework.
"""))

# ── Install ─────────────────────────────────────────────────────────────────
cells.append(code(r"""# Install dependencies if needed
# %pip install tiktoken matplotlib numpy
"""))

cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import json
"""))

# ── Section 1: Tokenization ──────────────────────────────────────────────────
cells.append(md(r"""## Section 1 — Tokenization in Practice

Before a model sees any text it sees tokens — subword units produced by Byte-Pair Encoding (BPE). Claude uses a similar approach to OpenAI's `tiktoken`. The same information encoded differently can cost very different amounts of tokens.
"""))

cells.append(code(r"""enc = tiktoken.get_encoding("cl100k_base")

inputs = {
    "Plain English":   "The quarterly revenue increased by twelve percent compared to last year.",
    "Same info as JSON": '{"metric": "quarterly_revenue", "change": 0.12, "period": "year-over-year"}',
    "Python code":     'revenue_change = (current_revenue - previous_revenue) / previous_revenue',
    "Spanish":         "Los ingresos trimestrales aumentaron un doce por ciento en comparacion con el ano anterior.",
    "URL":             "https://api.example.com/v2/financials/revenue?period=quarterly&compare=yoy&format=json",
}

print(f"{'Input Type':<20} {'Chars':>6} {'Tokens':>7} {'Chars/Token':>12}")
print("-" * 50)

for label, text in inputs.items():
    token_ids = enc.encode(text)
    chars = len(text)
    toks = len(token_ids)
    ratio = chars / toks
    print(f"{label:<20} {chars:>6} {toks:>7} {ratio:>12.2f}")

print()
print("Token detail for JSON input:")
json_ids = enc.encode(inputs["Same info as JSON"])
print([enc.decode([t]) for t in json_ids])
"""))

cells.append(md(r"""**Why this matters for API cost:** Every brace, colon, and quote in a JSON payload is a separate token. When you pass raw JSON to Claude as part of a system prompt or tool response, you're paying 2–3× more tokens than if you passed the same information as plain prose. The first optimization when a client says "our API costs are too high" is to token-count the inputs — before changing anything else.
"""))

# ── Section 2: Embeddings ────────────────────────────────────────────────────
cells.append(md(r"""## Section 2 — Embeddings and Semantic Geometry

Embeddings map words into high-dimensional vectors where semantic relationships become geometric ones. Words with similar meaning end up close together. Relationships like gender and royalty become consistent directions in the space.

We'll simulate a small 4-dimensional embedding space with 6 words and hand-craft vectors that encode interpretable dimensions: `[gender, royalty, geography, abstraction]`.
"""))

cells.append(code(r"""# 4D embeddings: [gender(F=1,M=-1), royalty, geography(Europe), abstraction]
embeddings = {
    "king":   np.array([-1.0,  1.0,  0.0,  0.5]),
    "queen":  np.array([ 1.0,  1.0,  0.0,  0.5]),
    "man":    np.array([-1.0,  0.0,  0.0,  0.0]),
    "woman":  np.array([ 1.0,  0.0,  0.0,  0.0]),
    "paris":  np.array([ 0.0,  0.0,  1.0, -0.5]),
    "france": np.array([ 0.0,  0.0,  1.0,  0.5]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

words = list(embeddings.keys())
n = len(words)
sim_matrix = np.zeros((n, n))

for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        sim_matrix[i, j] = cosine_similarity(embeddings[w1], embeddings[w2])

print("Cosine similarity matrix:")
print(f"{'':8}", end="")
for w in words:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:8}", end="")
    for j in range(n):
        print(f"{sim_matrix[i,j]:8.2f}", end="")
    print()
"""))

cells.append(code(r"""# Classic analogy: king - man + woman ≈ queen
result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

print("king - man + woman =", result_vec)
print()
print("Similarity to each word in space:")
for word, vec in embeddings.items():
    sim = cosine_similarity(result_vec, vec)
    print(f"  {word:8}: {sim:.4f}")

closest = max(embeddings.keys(), key=lambda w: cosine_similarity(result_vec, embeddings[w]))
print(f"\nClosest word: '{closest}'")
"""))

cells.append(md(r"""Real embeddings are 4,096–8,192 dimensions learned entirely from training data — nobody hand-crafts them. The geometry is the same. Semantic relationships become spatial relationships, and arithmetic in that space corresponds to analogy reasoning. This is why RAG retrieval works: embedding-based similarity search finds semantically related content, not just keyword matches.
"""))

# ── Section 3: Attention from scratch ───────────────────────────────────────
cells.append(md(r"""## Section 3 — Attention from Scratch

The core of a transformer is scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
```

- **Q** (Query): what each token is looking for
- **K** (Key): what each token offers
- **V** (Value): what information each token passes forward
- **sqrt(d_k)**: scaling factor to prevent extreme dot products that would collapse softmax

We'll implement this in pure NumPy and show the attention weight matrix for a 4-token sequence.
"""))

cells.append(code(r"""def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, verbose=False):
    d_k = Q.shape[-1]
    scores = Q @ K.T          # (seq_len, seq_len)
    scaled = scores / np.sqrt(d_k)
    weights = softmax(scaled)  # attention weight matrix
    output = weights @ V
    if verbose:
        print(f"d_k = {d_k}, scaling factor = sqrt({d_k}) = {np.sqrt(d_k):.3f}")
        print(f"Raw scores range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"Scaled scores range: [{scaled.min():.3f}, {scaled.max():.3f}]")
    return output, weights

# 4-token sequence: ["The", "trophy", "wont", "fit"]
# Hand-crafted Q/K/V to show "trophy" attending strongly to "fit"
np.random.seed(7)
seq_len, d_k = 4, 3
tokens = ["The", "trophy", "wont", "fit"]

Q = np.array([
    [0.1,  0.2,  0.0],   # The
    [0.9,  0.1,  0.8],   # trophy  — seeks size/constraint info
    [0.2,  0.5,  0.1],   # wont
    [0.8,  0.2,  0.9],   # fit     — offers size/constraint info
])
K = np.array([
    [0.0,  0.1,  0.2],
    [0.3,  0.0,  0.4],
    [0.1,  0.6,  0.0],
    [0.9,  0.1,  0.8],   # fit key matches trophy query
])
V = np.array([
    [0.1,  0.0,  0.2],
    [0.5,  0.8,  0.3],
    [0.2,  0.1,  0.4],
    [0.7,  0.3,  0.9],
])

output, weights = scaled_dot_product_attention(Q, K, V, verbose=True)

print("\nAttention weight matrix (rows=query token, cols=key token):")
print(f"{'':10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()
for i, t in enumerate(tokens):
    print(f"{t:10}", end="")
    for j in range(len(tokens)):
        print(f"{weights[i,j]:10.3f}", end="")
    print()
"""))

cells.append(code(r"""# Visualize as heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticklabels(tokens, fontsize=12)
ax.set_xlabel("Key (attends to)", fontsize=12)
ax.set_ylabel("Query (from)", fontsize=12)
ax.set_title("Attention weights", fontsize=13)
for i in range(len(tokens)):
    for j in range(len(tokens)):
        ax.text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center",
                color="white" if weights[i,j] > 0.5 else "black", fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=120)
plt.show()
print("Saved attention_heatmap.png")
"""))

cells.append(code(r"""# What happens without the scaling factor?
print("=== Effect of removing sqrt(d_k) scaling ===\n")

scores_unscaled = Q @ K.T
weights_unscaled = softmax(scores_unscaled)

scores_scaled = Q @ K.T / np.sqrt(d_k)
weights_scaled = softmax(scores_scaled)

print("Without scaling — attention weights for 'trophy':")
for t, w in zip(tokens, weights_unscaled[1]):
    print(f"  {t:8}: {w:.4f}")

print("\nWith scaling — attention weights for 'trophy':")
for t, w in zip(tokens, weights_scaled[1]):
    print(f"  {t:8}: {w:.4f}")

print(f"\nMax weight without scaling: {weights_unscaled.max():.4f}")
print(f"Max weight with scaling:    {weights_scaled.max():.4f}")
print("\nWithout scaling, softmax collapses toward one-hot — winner-take-all attention.")
print("The model loses the ability to blend information from multiple tokens.")
"""))

cells.append(md(r"""The scaling factor `sqrt(d_k)` is not a cosmetic detail. As embedding dimensions grow (d_k = 64, 128, 512...), the raw dot products grow in magnitude, pushing softmax into regions where gradients vanish. Without scaling, attention becomes winner-take-all: one token captures nearly all the weight and the rest disappear. The model can no longer blend information from multiple positions.
"""))

# ── Section 4: Multi-head attention ─────────────────────────────────────────
cells.append(md(r"""## Section 4 — Multi-Head Attention Intuition

A single attention head looks at the sequence through one lens. Multi-head attention runs several heads in parallel, each with its own Q/K/V projection matrices. After attending, outputs are concatenated and projected back down.

The key insight: heads are not manually assigned roles. During training the model discovers that specialization is useful, and different heads converge on different patterns — syntactic dependencies, semantic relationships, positional proximity.
"""))

cells.append(code(r"""np.random.seed(42)

def attention_head(Q_proj, K_proj, V_proj):
    _, weights = scaled_dot_product_attention(Q_proj, K_proj, V_proj)
    output = weights @ V_proj
    return output, weights

# Two heads with different random projection matrices
d_model = 3  # same as our Q/K/V dimension above
d_head = 3

W_Q1 = np.random.randn(d_model, d_head) * 0.5
W_K1 = np.random.randn(d_model, d_head) * 0.5
W_V1 = np.random.randn(d_model, d_head) * 0.5

W_Q2 = np.random.randn(d_model, d_head) * 0.5
W_K2 = np.random.randn(d_model, d_head) * 0.5
W_V2 = np.random.randn(d_model, d_head) * 0.5

# Project input sequence through each head
X = Q  # reuse our 4-token sequence as input

out1, w1 = attention_head(X @ W_Q1, X @ W_K1, X @ W_V1)
out2, w2 = attention_head(X @ W_Q2, X @ W_K2, X @ W_V2)

print("Head 1 attention weights:")
print(f"{'':10}", end="")
for t in tokens: print(f"{t:>10}", end="")
print()
for i, t in enumerate(tokens):
    print(f"{t:10}", end="")
    for j in range(len(tokens)): print(f"{w1[i,j]:10.3f}", end="")
    print()

print("\nHead 2 attention weights:")
print(f"{'':10}", end="")
for t in tokens: print(f"{t:>10}", end="")
print()
for i, t in enumerate(tokens):
    print(f"{t:10}", end="")
    for j in range(len(tokens)): print(f"{w2[i,j]:10.3f}", end="")
    print()

# Concatenate outputs
multi_head_output = np.concatenate([out1, out2], axis=-1)
print(f"\nConcatenated output shape: {multi_head_output.shape}")
print("(In a real transformer this is projected back to d_model via W_O)")
"""))

cells.append(md(r"""Each head sees the same tokens but through different learned projections, so it attends to different relationships. In practice, research has shown that some heads in trained models track subject-verb agreement, others track coreference, others track positional proximity. The model discovers this structure — nobody programs it in.
"""))

# ── Section 5: Quadratic scaling ─────────────────────────────────────────────
cells.append(md(r"""## Section 5 — Context Window and Quadratic Scaling

Attention computation scales as O(n²) with sequence length. Doubling the context doubles the number of tokens — and quadruples the attention computation. This is the hard constraint behind context window limits and the architectural motivation for RAG.
"""))

cells.append(code(r"""seq_lengths = [100, 1000, 10_000, 100_000, 200_000]
# Proportional cost: n^2 for attention, n for linear (RNN-style)
attention_cost = [n**2 for n in seq_lengths]
linear_cost    = [n    for n in seq_lengths]

# Normalize to seq_len=100 for readability
base = attention_cost[0]
attention_norm = [c / base for c in attention_cost]
linear_norm    = [c / base for c in linear_cost]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(seq_lengths, attention_norm, "b-o", label="Attention — O(n²)", linewidth=2)
ax.plot(seq_lengths, linear_norm,    "g--s", label="Linear — O(n)", linewidth=2)

# Mark Claude's 200K context window
claude_cost = (200_000**2) / base
ax.axvline(x=200_000, color="orange", linestyle=":", linewidth=1.5)
ax.annotate("Claude 200K\ncontext window",
            xy=(200_000, claude_cost * 0.15),
            fontsize=9, color="darkorange", ha="right")

ax.set_xlabel("Sequence length (tokens)", fontsize=12)
ax.set_ylabel("Relative compute cost (normalized to 100 tokens)", fontsize=11)
ax.set_title("Attention compute cost — quadratic vs linear scaling", fontsize=13)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("quadratic_scaling.png", dpi=120)
plt.show()

print("Compute cost relative to 100-token baseline:")
print(f"{'Seq length':>12} {'Attention (n^2)':>18} {'Linear (n)':>12}")
print("-" * 45)
for sl, ac, lc in zip(seq_lengths, attention_norm, linear_norm):
    print(f"{sl:>12,} {ac:>18,.0f}x {lc:>12,.0f}x")
"""))

cells.append(md(r"""RAG is the engineering workaround for this curve. Rather than extending context windows indefinitely (quadratic cost), you index your knowledge base as embeddings, retrieve the 2–3K most relevant tokens at query time, and inject only those into context. You convert an O(n²) problem into an O(k²) problem where k is small and fixed. For 10M tokens of documentation, the cost difference is roughly 10 billion to one.
"""))

# ── Section 6: Cost calculator ───────────────────────────────────────────────
cells.append(md(r"""## Section 6 — Tokenization Cost Calculator

A practical tool: estimate the API cost of different input types before sending them. The first question when API costs are high should always be "what are we actually sending?"

Prices below are approximate and subject to change — check the Anthropic pricing page for current rates.
"""))

cells.append(code(r"""PRICING = {
    "claude-sonnet": {"input": 3.0,  "output": 15.0},  # $ per million tokens (approximate)
    "claude-haiku":  {"input": 0.25, "output": 1.25},
    "claude-opus":   {"input": 15.0, "output": 75.0},
}

def estimate_api_cost(text, model="claude-sonnet", direction="input"):
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(text))
    price_per_million = PRICING[model][direction]
    cost_usd = (token_count / 1_000_000) * price_per_million
    return token_count, cost_usd

test_inputs = {
    "Short user message":   "What is my current portfolio allocation?",
    "Long JSON payload":    json.dumps({
        "portfolio": [
            {"ticker": "AAPL", "shares": 50, "avg_cost": 145.20, "current_price": 189.34},
            {"ticker": "MSFT", "shares": 30, "avg_cost": 280.10, "current_price": 415.22},
            {"ticker": "GOOGL", "shares": 10, "avg_cost": 130.50, "current_price": 172.88},
            {"ticker": "AMZN", "shares": 20, "avg_cost": 105.75, "current_price": 193.45},
            {"ticker": "NVDA", "shares": 15, "avg_cost": 220.40, "current_price": 875.39},
        ],
        "cash_usd": 2847.63,
        "total_value_usd": 48293.11,
        "last_updated": "2026-05-13T09:30:00Z",
    }),
    "Python code snippet":  '''def calculate_portfolio_return(positions, cash):
    total_cost = sum(p["shares"] * p["avg_cost"] for p in positions)
    total_value = sum(p["shares"] * p["current_price"] for p in positions) + cash
    return (total_value - total_cost) / total_cost''',
    "System prompt":        'You are FinMentor, an educational financial guidance tool. You help users understand their portfolio composition, explain financial concepts, and provide general investment education. You do not provide personalized investment advice. Always clarify that users should consult a licensed financial advisor before making investment decisions. Respond in a clear, educational tone. When discussing returns or performance, explain the underlying calculation. Do not recommend specific trades or timing decisions.',
}

print(f"{'Input Type':<22} {'Tokens':>7} {'Cost (input, Sonnet)':>22} {'Cost (input, Haiku)':>21}")
print("-" * 75)
for label, text in test_inputs.items():
    toks, cost_sonnet = estimate_api_cost(text, "claude-sonnet", "input")
    _,    cost_haiku  = estimate_api_cost(text, "claude-haiku",  "input")
    print(f"{label:<22} {toks:>7} ${cost_sonnet:>20.6f} ${cost_haiku:>19.6f}")

print("\nNote: costs are per individual call. Multiply by daily call volume for real-world impact.")
print("Prices are approximate — verify at anthropic.com/pricing.")
"""))

cells.append(code(r"""# Show: compressed JSON prose vs raw JSON
raw_json = json.dumps({
    "portfolio": [
        {"ticker": "AAPL", "shares": 50, "avg_cost": 145.20, "current_price": 189.34},
        {"ticker": "MSFT", "shares": 30, "avg_cost": 280.10, "current_price": 415.22},
    ]
})
prose_equivalent = (
    "Portfolio: 50 shares AAPL (cost $145.20, now $189.34), "
    "30 shares MSFT (cost $280.10, now $415.22)."
)

toks_json,  _ = estimate_api_cost(raw_json, "claude-sonnet", "input")
toks_prose, _ = estimate_api_cost(prose_equivalent, "claude-sonnet", "input")

print(f"Raw JSON:  {toks_json} tokens  — '{raw_json[:80]}...'")
print(f"Prose:     {toks_prose} tokens  — '{prose_equivalent}'")
print(f"\nToken reduction: {toks_json - toks_prose} tokens ({(toks_json-toks_prose)/toks_json*100:.0f}% saved)")
"""))

cells.append(md(r"""This is why input formatting is a first-order engineering concern, not an afterthought. At scale, the difference between raw JSON and prose-formatted context can cut input costs by 30–50% with no change to model behavior.
"""))

# ── Section 7: Key observations ──────────────────────────────────────────────
cells.append(md(r"""## Section 7 — Key Observations

**1. JSON costs 2–3× more tokens than prose for equivalent information.**
Every brace, colon, and quote is a token. Compress structured data before sending it as context. When API costs spike, token-count the inputs before changing anything else.

**2. Attention weights are not uniform — token position matters.**
The model attends less reliably to content buried in the middle of long contexts. Critical instructions and key information should be placed near the beginning or end of context, not in the center of a long system prompt. This is the mechanism behind "lost-in-the-middle" failures.

**3. The scaling factor sqrt(d_k) is not optional.**
Without it, softmax collapses to near one-hot — winner-take-all attention where a single token captures nearly all the weight. The model loses the ability to synthesize information from multiple positions. This is not a training detail; it's a mathematical requirement of the architecture.

**4. RAG is the engineering solution to the quadratic scaling constraint.**
Context windows have hard limits because attention cost grows as O(n²). RAG converts a potentially unbounded retrieval problem into a fixed-cost injection: retrieve k relevant tokens, attend over k² pairs instead of n² pairs. The technique exists because of this equation, not despite it.
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = os.path.join(os.path.dirname(__file__), "transformer_fundamentals.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out}")
