# Lesson 7 — Speculative Decoding
> Curriculum: Anthropic FDE Prep
> Date: 2026-06-29
> Status: draft

---

## What this lesson is about

Large language models generate one token at a time, and each token requires a full forward pass through the entire model — even when the next word is obvious. Speculative decoding breaks that bottleneck: a small, fast draft model proposes several tokens at once, and the large target model verifies all of them in a single parallel pass. Because verification is cheaper than generation, and because the acceptance criterion is mathematically exact, you get the target model's output distribution at meaningfully higher throughput — with no quality tradeoff.

---

## Core concepts

### The autoregressive bottleneck

Standard transformer decoding generates tokens one at a time. Each step samples from `P(x_t | x_{<t})`, runs a full forward pass, and then feeds the new token back as context for the next step. This is inherently sequential — you cannot generate token `t+1` before you have token `t`. On modern hardware, this makes LLM inference memory-bandwidth-bound rather than compute-bound: the GPU is mostly waiting for weights to be loaded, not doing heavy matrix multiplication.

**Technical depth:**
The KV cache (Lesson 3) already addresses one part of this: past attention keys and values are cached so each new token only needs to attend to cached entries, reducing per-step compute from `O(n²)` to `O(n)`. But the KV cache does nothing about the sequential dependency — you still need `N` serial forward passes to generate `N` tokens. Speculative decoding attacks the sequential dependency directly. It uses the observation that the target model's forward pass cost does not scale linearly with sequence length for short extensions — running the target on `prompt + K draft tokens` costs roughly the same as running it on `prompt + 1 token`, because the bottleneck is memory bandwidth for loading weights, not the attention computation.

---

### Draft-then-verify mechanism

A small draft model generates K tokens greedily. The large target model then verifies all K tokens in one forward pass, accepting or rejecting each based on how much its probability for that token differs from the draft's. The output distribution is mathematically identical to sampling from the target model alone — making speculative decoding provably lossless.

**Technical depth:**
The acceptance criterion for draft token `d_i` is: accept with probability `min(1, p_target(d_i) / p_draft(d_i))`. If rejected at position `j`, sample a corrected token from the adjusted distribution `p_target - p_draft` (clipped to non-negative), then discard all drafts at positions `j+1...K`. This rejection sampling scheme guarantees that the joint distribution over accepted tokens equals the target model's distribution exactly — not approximately. Expected new tokens per step: `E[tokens] = (1 - α^{K+1}) / (1 - α)`, where `α` is the per-token acceptance rate. At `α = 0.85` and `K = 4`, that is ~3.2 tokens per target forward pass instead of 1.

---

### Choosing the draft model

The draft model must be fast enough that K draft forward passes cost less than one saved target forward pass, and accurate enough that the acceptance rate α is high. Both conditions point toward the same architecture family at smaller scale.

**Technical depth:**
The same tokenizer is required — token IDs must align between draft and target. Within-family models (Qwen2.5-0.5B drafting for Qwen2.5-3B) outperform cross-family pairs because the models share vocabulary and have correlated internal representations. EAGLE-3 (2024) takes this further: instead of a separate small model, it trains a lightweight draft head directly on the target model's hidden states, achieving `α > 0.90` on coding tasks. The speedup ceiling is hardware-dependent: at batch size 1 (typical in agent pipelines), the target forward pass is bandwidth-bound and scales poorly with sequence length, so spec decode wins easily. At large batch sizes, the target is already compute-saturated and the benefit shrinks.

---

### When it helps vs. when it doesn't

Speculative decoding helps most when outputs are predictable — structured formats like JSON, SQL, and formatted markdown tables have high acceptance rates because the draft model can anticipate what the target would say. Open-ended narrative text has lower α, reducing the benefit. The cost: both models must fit in VRAM simultaneously.

**Technical depth:**
Task type correlates strongly with α. Structured extraction and code generation tend to `α ≈ 0.80–0.90` because token sequences are constrained by format. Templated tables fall in the `α ≈ 0.70–0.80` range. Free-form narrative is `α ≈ 0.55–0.65`. At `α = 0.60`, `K = 4`, expected tokens per step is ~1.97 — barely faster than standard decoding, and the overhead of loading the draft model may cancel the gain. The practical decision: profile acceptance rate per task type before committing to a spec decode deployment. This notebook builds that profiler.

---

## How it connects to prior lessons

- **Lesson 3 — Transformers:** The KV cache and speculative decoding are complementary optimizations. The KV cache reduces per-step compute cost from `O(n²)` to `O(n)`. Speculative decoding reduces the number of sequential steps by batching verification. In production, you deploy both simultaneously — the target model runs with its KV cache, and the draft model proposes ahead of it.
- **Lesson 6 — Fine-tuning & PEFT:** LoRA-trained smaller models are excellent draft candidates. A Qwen2.5-0.5B model fine-tuned on M&A diligence output formats (using Lesson 6 techniques) would have higher `α` against a Qwen2.5-3B target than the base 0.5B model, because fine-tuning aligns the draft's probability distribution more closely with the target's on that specific task. The Lesson 6 demand classifier notebook is the prototype for this workflow.

---

## The Anthropic angle

Speculative decoding is one of the rare inference optimizations that is provably lossless. Most techniques that increase throughput — quantization, pruning, distillation — degrade output quality to some degree. Speculative decoding changes *how* the target model's distribution is sampled, not *what* that distribution is. This is why it can be applied transparently inside Claude's inference stack without changing the quality bar that Anthropic guarantees.

The FDE-relevant insight is architectural: speculative decoding benefits are not uniform across sub-agents. In the M&A Diligence Intelligence System, the financial parsing sub-agent (structured JSON extraction) and the benchmarking sub-agent (formatted markdown tables) are strong candidates. The IC memo drafting sub-agent (open-ended narrative) is a weaker candidate. An FDE advising on inference optimization needs to profile acceptance rates per task type before recommending spec decode — which is exactly what this lesson's notebook does.

---

## Hands-on project

**Title:** Speculative Decoding Acceptance Rate Profiler

**What you build:** A notebook that runs speculative decoding across three M&A Diligence task types (JSON extraction, markdown table generation, IC memo drafting) and measures per-task and per-position acceptance rates — producing the evidence base for which sub-agents benefit from spec decode deployment.

**Connection to M&A Diligence Intelligence System:** The three task types directly mirror the three highest-volume M&A sub-agents. The profiler's output — measured α per sub-agent — is the input to the deployment decision: which sub-agents justify the VRAM cost of holding both draft and target models simultaneously.

**Steps:**
1. Generate 60 synthetic prompts (20 per task type) using XML-structured templates in `prompts.py`
2. Load Qwen2.5-0.5B as draft model; Qwen2.5-3B as target (GPU) or simulate (CPU)
3. Implement the speculative decoding loop: draft K=4 tokens, verify with target, record per-position acceptance
4. Measure α by task type; compute theoretical speedup `(1 - α^{K+1}) / (1 - α)`
5. Plot: acceptance rate bar chart, speedup overlay, token-position acceptance curve
6. Print decision: which sub-agent benefits most from spec decode, with the measured α as evidence

**Deliverable:** `notebook/notebook.ipynb` with all charts saved to `07-speculative-decoding/charts/`

---

## Key terms

**`α` (acceptance rate):** The per-token probability that the draft model's proposed token matches what the target model would have chosen, used to compute expected speedup.

**Draft model:** The smaller, faster model that proposes K tokens per speculation step; must share the tokenizer with the target.

**Target model:** The large model whose output distribution is preserved exactly; verified in one parallel forward pass per speculation step.

**Lossless decoding:** The property that speculative decoding's output distribution is identical to standard autoregressive sampling from the target model — no quality tradeoff.

**EAGLE / EAGLE-3:** A speculative decoding variant that trains a lightweight draft head directly on the target model's hidden states, achieving higher α than a standalone small model.

**Theoretical speedup:** `E[tokens per step] = (1 - α^{K+1}) / (1 - α)`; at α=0.85, K=4 this is ~3.2×.

**Memory bandwidth bound:** The hardware regime where inference cost is dominated by loading model weights from HBM rather than matrix multiply compute — the regime where speculative decoding yields the largest gains.

---

## Further reading

- **Leviathan et al. (2022) — "Fast Inference from Transformers via Speculative Decoding"** (ICML 2023): The original paper; Section 3 proves the losslessness guarantee — read it to understand why rejection sampling preserves the target distribution exactly, not just approximately.
- **Li et al. (2024) — EAGLE-3** ([arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)): The current state-of-the-art draft head approach; read the ablation section to understand why training on hidden states beats a standalone small model.
- **PyTorch Blog — "Hitchhiker's Guide to Speculative Decoding"**: A practical implementation walkthrough with benchmarks across task types — directly relevant to calibrating expectations for M&A diligence task categories.
- **Anthropic Engineering Blog — Inference optimization**: Anthropic's public writing on efficiency, relevant for understanding which optimizations are compatible with constitutional training objectives.

---

## Open questions for next session

1. After running the profiler: which task type had the highest `α`? How does the measured speedup compare to the theoretical formula? What explains the gap (if any)?
2. The losslessness proof assumes exact arithmetic. In practice, models use fp16/bf16. Does quantization break the guarantee? What would you test to verify it doesn't?
3. If you were to deploy spec decode for the financial parsing sub-agent in M&A Diligence: what draft model would you choose, and how would you fine-tune it (using Lesson 6 techniques) to maximize α specifically for JSON extraction from financial statements?
