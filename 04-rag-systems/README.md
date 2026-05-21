# 04 — RAG Systems

**Date:** May 2026
**Paper:** Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)
**Status:** Complete

## What's Here

| File | Description |
|------|-------------|
| `notebook/rag_fundamentals.ipynb` | Working notebook: chunking, vector index, query routing, RAGAS eval, freshness demo |
| `notebook/rag_advanced_retrieval.ipynb` | Advanced retrieval — hybrid search, HyDE, re-ranking, query decomposition |
| `notebook/rag_evaluation.ipynb` | RAG evaluation — golden datasets, RAGAS, LLM-as-judge, MLflow tracking |
| `notebook/graphrag_and_advanced_architectures.ipynb` | GraphRAG, corrective RAG, agentic RAG |
| `POST.md` | LinkedIn post draft |

## Core Idea

RAG solves the context window constraint and knowledge cutoff problem by embedding documents into a vector index, retrieving only semantically relevant chunks at query time via cosine similarity, and injecting those chunks into the generation context. Three stages: **index**, **retrieve**, **generate**.

## Key Concepts

- **Chunking strategies:** fixed-size, semantic, hierarchical — chunking quality determines retrieval quality more than embedding model choice
- **Vector stores:** ChromaDB, Pinecone, Weaviate — store embeddings alongside metadata for filtered retrieval
- **Cosine similarity:** the distance metric used to rank retrieved chunks against the query embedding
- **Top-K retrieval:** return the K most similar chunks; K is a tunable hyperparameter trading cost vs. context
- **Query routing:** a cheap classification call that decides which index(es) to search before retrieval
- **Multi-index retrieval:** separate indexes for different data types (portfolio data vs. news); route queries to the right one
- **Index freshness:** stale indexes produce confident but wrong answers — correctness bug, not a performance bug
- **Vocabulary mismatch:** query terms that don't match chunk terms even when semantically equivalent; mitigated by dense retrieval
- **Faithfulness:** did the answer come from retrieved context, not model training data?
- **Context recall:** did retrieval surface the chunks that actually contain the answer?
- **Answer relevance:** did the final answer address what the user asked?
- **RAGAS:** evaluation framework measuring all three metrics on a golden dataset

## Why RAG Exists

RAG solves three constraints simultaneously:

1. **O(n²) attention scaling** — full-context retrieval over a large document corpus is computationally prohibitive; RAG retrieves only relevant chunks
2. **Knowledge cutoff** — LLMs have a training cutoff; RAG indexes live data, making the model's effective knowledge current
3. **Noisy full-context retrieval** — stuffing everything into context degrades answer quality; RAG injects only the relevant signal

## FDE Relevance — FinMentor Architecture Diagnosis

- **Same snapshot regardless of question:** FinMentor today sends the full IBKR portfolio dump for every query — this is the naive approach RAG replaces
- **Index freshness bug:** real-time price data must never be indexed; it must be fetched fresh at query time and injected directly
- **Critic agent maps to faithfulness metric:** the critic checks whether the answer came from grounded data — that is faithfulness in RAGAS terms
- **Multi-index needed:** portfolio position queries and market news queries should hit separate indexes and be routed before retrieval

## What I'd Build Next

Add a RAGAS evaluation harness that measures **faithfulness**, **context recall**, and **answer relevance** on a fixed golden dataset of 20 FinMentor-style questions. Run it nightly against the production RAG pipeline to catch regressions when the index schema or embedding model changes.
