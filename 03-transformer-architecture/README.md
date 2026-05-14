# 03 — Transformer Architecture

**Date:** May 2026
**Paper:** Vaswani et al., *Attention Is All You Need* (2017)
**Status:** Complete

## What's Here

| File | Description |
|------|-------------|
| [notebook/transformer_fundamentals.ipynb](notebook/transformer_fundamentals.ipynb) | Tokenization cost, embeddings, attention from scratch, quadratic scaling visualization |

## Core Idea

Transformers replaced sequential RNN processing with parallel attention — every token attends to every other token simultaneously via Query, Key, and Value vectors. Each token computes a query (what am I looking for?), keys (what do I offer?), and values (what information do I carry?). The dot product of queries against all keys, scaled by sqrt(d_k) and passed through softmax, produces an attention weight matrix that determines how much each token draws from every other. This runs in parallel across all tokens simultaneously. Stacked across 80–100+ layers with feed-forward networks between attention blocks — the attention layers route information, the feed-forward layers transform and store it. Context window is bounded by the quadratic scaling of attention computation: doubling sequence length quadruples compute cost.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Tokenization** | Text split into subword units via BPE; token count ≠ word count — JSON and code tokenize more expensively than prose |
| **Embeddings** | High-dimensional vectors where semantic relationships become spatial relationships — similar meaning, proximate vectors |
| **Self-attention** | Each token attends to all others simultaneously; captures long-range dependencies RNNs couldn't |
| **Q/K/V** | Query (what to look for), Key (what each token offers), Value (what information to pass forward) |
| **Multi-head attention** | Multiple attention heads in parallel; each learns to specialize on different relationship types |
| **Feed-forward networks** | Two-layer MLP applied identically to each position; where factual knowledge is stored |
| **Residual connections** | Each sublayer adds its output to its input — enables training very deep networks |
| **Layer norm** | Normalizes activations before each sublayer — stabilizes training |
| **Context window** | Maximum sequence length the model can attend over — bounded by quadratic compute cost |
| **Quadratic scaling** | Attention cost grows as O(n²) with sequence length — 2× tokens = 4× compute |
| **Lost-in-the-middle** | Model attends less reliably to content in the center of long contexts vs. beginning and end |

## FDE Relevance

- **JSON tokenization cost:** Passing raw JSON payloads is expensive — braces, quotes, and colons each consume tokens. Compressing to plain prose or compacting JSON before sending can cut input token count by 2–3×.
- **Context window limits and RAG:** The quadratic scaling constraint is why RAG exists as an engineering pattern. You retrieve the 2–3K relevant tokens and inject only those — instead of fitting all source documents into context.
- **Prompt position effects:** Instructions placed at the start and end of the context window are followed more reliably than those buried in the middle. Long system prompts with critical rules in the center are a reliability risk.
- **Feed-forward networks as knowledge store:** Factual knowledge lives in the feed-forward weights, not in the attention mechanism. When a model "knows" that Paris is the capital of France, that knowledge is encoded in the FFN layers — which is why RAG works: you're routing around stale or missing FFN knowledge by injecting the answer into context.

## What I'd Build Next

Implement attention from scratch in pure NumPy — no frameworks, just the math — to verify that the QK^T/sqrt(d_k) formula produces sensible attention distributions on a real sentence. Then add positional encodings to confirm that without them, token order is invisible to the model. That's the experiment that makes "position matters" concrete rather than theoretical.
