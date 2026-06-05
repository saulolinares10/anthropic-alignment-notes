# Lesson 6 — Fine-tuning & PEFT
> Curriculum: Anthropic FDE Prep
> Date: 2026-06-05
> Status: draft

---

## What this lesson is about

Pre-trained language models know a lot about language but nothing specific about your task. Fine-tuning is the process of continuing to train a model on your own data so it behaves the way you need — in a specific tone, domain, or format. Parameter-Efficient Fine-Tuning (PEFT) solves the practical problem that full fine-tuning is expensive: methods like LoRA let you adapt a model by training only a tiny fraction of its weights, getting most of the benefit at a fraction of the cost.

---

## Core concepts

### Fine-tuning vs. prompting vs. RAG

Before touching any training code, the most important skill is knowing when fine-tuning is the wrong answer. Prompting changes what you ask the model. RAG changes what information the model sees. Fine-tuning changes the model itself. Each solves a different problem.

Prompting is the right default — it's free, reversible, and fast to iterate. RAG is the right answer when the model needs access to facts it wasn't trained on (documents, databases, real-time data). Fine-tuning earns its place only when you need the model to consistently adopt a style, format, or behavior that prompting can't reliably produce, or when you're doing something at such high volume that a smaller fine-tuned model is cheaper than repeated Sonnet calls.

**Technical depth:**
The decision framework in production: (1) Can prompting achieve this? If yes, stop. (2) Is this a knowledge problem (the model doesn't know the facts)? Use RAG. (3) Is this a behavior problem (the model knows how but won't do it consistently, or the output format is rigid)? Consider fine-tuning. (4) Is the volume high enough that a fine-tuned smaller model's inference savings justify the training cost? If no, keep using the base model. Fine-tuning on bad data produces a model that confidently does the wrong thing — worse than a well-prompted base model.

---

### Supervised Fine-Tuning (SFT)

SFT is the simplest form: you take a pre-trained model and continue training it on (input, output) pairs — examples of exactly the behavior you want. The model updates its weights to make those outputs more probable given those inputs. This is the same mechanism used in Lesson 2's RLHF pipeline: SFT is Step 1 of InstructGPT, producing the initial model that reward modeling and PPO then refine.

The data requirement is the hardest part. You need examples that are high-quality, representative of the full distribution of inputs you'll see in production, and correctly labeled. A common failure mode: fine-tuning on 500 examples of one narrow behavior makes the model worse at everything else. This is called catastrophic forgetting — the model overwrites general capabilities while learning the specific ones.

**Technical depth:**
SFT minimizes cross-entropy loss on the target tokens: `L = -Σ log P(yᵢ | y<ᵢ, x)` where `x` is the input and `y` is the target output. Training runs for 1–3 epochs typically — more than that and the model memorizes rather than generalizes. Key hyperparameters: learning rate (usually 1e-5 to 5e-5, much lower than pre-training), batch size, and sequence length. Gradient checkpointing and mixed precision (bf16) are standard to fit in GPU memory. Dataset size rule of thumb: hundreds of examples for format/style adaptation, thousands for domain adaptation, tens of thousands for behavioral change at scale.

---

### LoRA — Low-Rank Adaptation

Full fine-tuning updates every weight in the model — for a 7B parameter model that's 7 billion floats to store and update. LoRA's insight is that the weight updates needed for most fine-tuning tasks are low-rank: they can be expressed as the product of two small matrices rather than one large one. Instead of updating weight matrix `W` directly, LoRA freezes `W` and adds `ΔW = A × B` where A and B are much smaller. During inference, `W + AB` behaves like a fine-tuned weight matrix. Only A and B are trained.

In practice this means you can fine-tune a 7B model on a single consumer GPU (24GB VRAM) instead of needing a multi-GPU cluster. The trained LoRA weights are a small file (tens of MB) rather than a full model copy.

**Technical depth:**
For a weight matrix `W ∈ ℝ^(d×k)`, LoRA introduces `A ∈ ℝ^(d×r)` and `B ∈ ℝ^(r×k)` where rank `r << min(d,k)`. The forward pass becomes `h = Wx + (AB)x × (α/r)` where `α` is a scaling hyperparameter. B is initialized to zero so LoRA starts as an identity transformation — no disruption to the pretrained model at initialization. Trainable parameters: `r×(d+k)` vs. `d×k` for full fine-tuning. At r=8 on a typical attention projection (d=k=4096), that's 65,536 parameters vs. 16,777,216 — a 256× reduction. LoRA is typically applied to the query and value projection matrices in attention layers. QLoRA adds 4-bit quantization of the frozen base weights, cutting memory further at a small quality cost.

---

### The fine-tune vs. RAG vs. prompt decision for Claude API users

This is the most practically relevant question for an FDE role. When you're building on Claude's API, you can't fine-tune Claude directly (Anthropic controls the model). Your tools are: prompt engineering, RAG, and — in limited cases — Claude's fine-tuning API for specific tiers. This reframes the lesson: understanding fine-tuning deeply lets you explain to a client *why* their use case doesn't need it, and design the right RAG or prompting architecture instead. When a client insists on fine-tuning, knowing the failure modes lets you scope the data collection and eval requirements honestly.

**Technical depth:**
For open-source models (Llama 3, Mistral, Phi) deployed on client infrastructure, LoRA fine-tuning is a real option. The workflow: base model → SFT with LoRA on domain data → merge LoRA weights → deploy. Evaluation must include both task-specific metrics and general capability regression tests — a fine-tuned model that scores 15% better on your task but 30% worse on general instructions is not a win. PEFT library from Hugging Face handles LoRA setup; `trl` handles SFT training loop with dataset formatting.

---

## How it connects to prior lessons

- **Lesson 2 — RLHF:** SFT is literally Step 1 of the InstructGPT pipeline. The reward model in Lesson 2 is trained on top of an SFT model, not the raw pretrained base. Fine-tuning is not an alternative to RLHF — it's the foundation it runs on.
- **Lesson 3 — Transformers:** LoRA targets the attention weight matrices (Q, K, V projections) specifically because those are where task-specific behavioral changes concentrate. Understanding which layers to target requires knowing what each layer does.
- **Lesson 4 — RAG:** The fine-tune vs. RAG decision is the most common architectural choice you'll face. RAG wins when the problem is missing knowledge; fine-tuning wins when the problem is missing behavior. They're not interchangeable.

---

## The Anthropic angle

Anthropic interviewers will probe whether you understand fine-tuning as a component of a larger system, not a standalone fix. The RLHF pipeline (Lesson 2) depends on SFT producing a stable, instruction-following base — if the SFT data is noisy or narrow, the reward model has nothing good to learn from. This is why Anthropic invests heavily in data quality and human feedback pipelines, not just training algorithms.

The sharper interview question is: "When would you advise a customer not to fine-tune?" The expected answer covers: (1) they haven't exhausted prompt engineering, (2) their problem is knowledge retrieval not behavior, (3) they don't have enough high-quality labeled data, (4) they can't afford ongoing evals to catch regression. A candidate who leads with "fine-tuning makes the model better" without discussing failure modes reads as junior. A candidate who can articulate the data quality bar and the regression risk reads as someone who has shipped this.

---

## Hands-on project

### Part A — LoRA fine-tuning script from scratch

**What you build:** A working LoRA fine-tuning notebook using Hugging Face PEFT + TRL on a small open-source model (Phi-3-mini or Llama-3.2-1B), trained on a small domain-specific dataset.

**Connection to M&A Diligence:** The IC memo drafting sub-agent currently relies on a prompted Sonnet call. This project builds the evidence base for whether a fine-tuned smaller model could replace it for structured memo sections — reducing cost at high volume while maintaining output format consistency.

**Steps:**
1. Set up environment: `pip install transformers peft trl datasets bitsandbytes`
2. Load a base model in 4-bit (QLoRA): `BitsAndBytesConfig` + `AutoModelForCausalLM`
3. Define LoRA config: `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])`
4. Prepare a small dataset of 100–200 (instruction, output) pairs in ChatML format — use anonymized PE memo snippets or synthetic examples, never production data
5. Run SFT with `SFTTrainer`, log loss curve
6. Generate 10 sample outputs from base model vs. fine-tuned model on held-out prompts
7. Manual eval: score each on format adherence, tone, and factual accuracy

**Deliverable:** `notebooks/lesson-06-lora-finetuning.ipynb` with loss curve, sample outputs, and a 3-sentence verdict on whether the fine-tuned model beats the base.

---

### Part B — Fine-tune vs. RAG vs. prompt comparison for M&A Diligence

**What you build:** A structured comparison notebook evaluating three approaches for the IC memo drafting sub-agent: zero-shot prompt, RAG-augmented prompt, and LoRA fine-tuned model.

**Connection to M&A Diligence:** This is a real architectural decision for the system. The result directly informs whether the IC memo sub-agent should use a fine-tuned open model or stay on prompted Sonnet.

**Steps:**
1. Define 20 test cases: (diligence input snippet → expected memo section) using synthetic/anonymized data
2. Implement Approach A: Sonnet with a well-engineered XML prompt (role/context/task/format)
3. Implement Approach B: Sonnet + RAG (retrieve 3 relevant memo examples from a vector store)
4. Implement Approach C: Part A's fine-tuned model (no retrieval)
5. Score all 20 outputs on: format adherence (0/1), key risk surfaced (0/1), hallucination present (0/1)
6. Record latency and estimated cost per call for each approach
7. Write a 1-page decision memo: recommended architecture with justification

**Deliverable:** `notebooks/lesson-06-finetuning-vs-rag-comparison.ipynb` + `decision-memo-ic-drafting.md`

---

## Key terms

**LoRA (Low-Rank Adaptation):** A PEFT method that trains two small matrices (A, B) added to frozen pretrained weights instead of updating the full weight matrix.

**QLoRA:** LoRA applied on top of a 4-bit quantized base model, reducing GPU memory requirements enough to fine-tune large models on consumer hardware.

**SFT (Supervised Fine-Tuning):** Training a model on (input, output) pairs to make specific outputs more probable — the first step of the RLHF pipeline.

**Catastrophic forgetting:** When fine-tuning on a narrow dataset degrades the model's general capabilities by overwriting broadly useful weights.

**Rank (r):** The bottleneck dimension in LoRA's AB decomposition — lower rank = fewer trainable parameters = less expressive adaptation. Typical values: 4, 8, 16.

**PEFT:** Parameter-Efficient Fine-Tuning — the family of methods (LoRA, prefix tuning, adapters) that adapt models by training far fewer parameters than full fine-tuning.

**Merge and unload:** The step after LoRA training where A and B are multiplied and added back into W, producing a standard model that needs no LoRA-specific inference code.

---

## Further reading

- **Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models** ([arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)): The original paper — read Section 4 (the rank analysis) to understand why low-rank works, not just that it works.
- **Hugging Face PEFT docs** ([huggingface.co/docs/peft](https://huggingface.co/docs/peft)): The practical reference for implementing LoRA, QLoRA, and other PEFT methods — use alongside the notebook in Part A.
- **Dettmers et al. (2023) — QLoRA** ([arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)): Read the memory analysis section — it explains exactly why 4-bit quantization + LoRA makes large model fine-tuning accessible and what quality you trade away.
- **Anthropic docs — When to fine-tune** ([docs.anthropic.com](https://docs.anthropic.com)): Anthropic's own guidance on when fine-tuning Claude is appropriate vs. prompt engineering — directly relevant to the FDE role and the framing interviewers expect.

---

## Open questions for next session

1. After running Part A: what rank `r` did you use, and how did output quality change when you halved it? What does that tell you about the task's intrinsic dimensionality?
2. After running Part B: which approach won on quality, and which won on cost? Were they the same? What would the break-even call volume be for fine-tuning to beat RAG on cost?
3. For the M&A Diligence IC memo sub-agent specifically: given your comparison results, what's your recommended architecture — and what data would you need to collect to make fine-tuning viable if the cost math eventually favored it?
