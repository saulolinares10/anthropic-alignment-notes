---
title: I finally understood why Claude refuses things. Here's what I found.
published: true
description: I was building a Claude-powered financial advisor and hit a wall I couldn't explain. It sent me to Anthropic's Constitutional AI paper — and what I found changed how I think about system prompts.
tags: ai, machinelearning, python, anthropic
cover_image: 
---

## The moment that made me want to understand this

I've been building FinMentor — a multi-agent financial advisor that runs on Claude. Four agents: a portfolio analyst, a market researcher, a macro economist, and a critic that reviews the others before the final answer goes out. It connects to my IBKR brokerage account. I use it daily.

One afternoon I ran a portfolio query — something like "how concentrated am I in tech, and should I be worried?" — and the response came back wrapped in so many caveats it was almost useless. The actual analysis was solid. But it was buried under three paragraphs of "this is not financial advice" and "it's important to consider your personal circumstances." I'd seen this before. I always blamed my system prompts.

So I rewrote them. Tighter, more direct, explicit instructions to be concise. Same pattern. I tried a completely different prompt structure. Still there.

That's when I stopped blaming my prompts. This wasn't coming from my instructions — it was somewhere deeper in the model. And I didn't actually know where.

That question sent me to Anthropic's 2022 paper: *Constitutional AI: Harmlessness from AI Feedback* by Bai et al.

## What Constitutional AI actually is (and what it isn't)

My initial mental model was wrong in a specific way. I assumed CAI was a rulebook — a list of prohibited outputs baked into the weights during fine-tuning. A very long system prompt the model couldn't override.

That's not it.

CAI is a training procedure in two phases.

**Phase 1 — SL-CAI (Supervised Learning):** You write a list of principles — the "constitution." The model generates a response to a prompt. Then you ask the *same model* to critique that response against one of the principles. Then you ask it to rewrite the response based on the critique. The (original prompt, rewritten response) pair becomes a supervised training example. No human annotator required.

**Phase 2 — RLAIF (Reinforcement Learning from AI Feedback):** Same mechanism applied to preference labeling. Instead of asking humans "which of these two responses is better?", you ask the AI — guided by the same constitution. That preference signal trains the reward model used for RL fine-tuning.

The key: RLHF at scale is bottlenecked by human annotation throughput. Each preference label requires real human attention. CAI breaks that bottleneck by using the model as its own judge. The cost of generating a preference label drops from "15 minutes of an annotator's time" to "one API call."

## What I built

I built a simulation of the SL-CAI loop using the Anthropic Python SDK. Three red-team prompts designed to elicit manipulation-adjacent responses, five constitutional principles, two revision cycles each. I logged every intermediate state — initial response, critique, revision — across all three prompts.

The most revealing function is `critique_response()`. This is the mechanical heart of CAI: the model evaluating its own output against a specific principle.

```python
def critique_response(prompt: str, response: str, principle: str) -> str:
    content = "\n\n".join([
        f"Evaluate this AI response against the principle: '{principle}'",
        f"User request: {prompt}",
        "Response to evaluate:",
        response,
        "Be concrete and specific: identify exact phrases that violate or could "
        "better align with the principle.",
    ])
    result = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You are a rigorous AI safety critic. Identify specific ways AI "
               "responses can be improved according to stated principles.",
        messages=[{"role": "user", "content": content}],
    )
    return result.content[0].text
```

And the full loop that chains generate → critique → revise:

```python
def run_cai_loop(prompt: str, n_cycles: int = 2, verbose: bool = True) -> dict:
    initial = generate_initial_response(prompt)
    cycles = []
    current = initial

    for i in range(n_cycles):
        principle = CONSTITUTION[i % len(CONSTITUTION)]
        critique = critique_response(prompt, current, principle)
        revised = revise_response(prompt, current, critique, principle)
        cycles.append({
            "cycle": i + 1, "principle": principle,
            "critique": critique, "revised": revised,
        })
        current = revised

    return {"prompt": prompt, "initial": initial, "cycles": cycles, "final": current}
```

The loop saves every intermediate state. That turned out to be the most interesting part of the whole experiment.

Full notebook: [https://github.com/saulolinares10/anthropic-alignment-notes](https://github.com/saulolinares10/anthropic-alignment-notes)

## What surprised me

**1. The first revision cycle does most of the work.** The delta between the initial response and the first revision was always significant. The delta between revision 1 and revision 2 was incremental — refinements, not transformations. If you're generating training data at scale, one cycle is probably sufficient. The law of diminishing returns hits fast.

**2. The same model plays both roles — and it actually works.** There's no separate critic model. The same Claude instance that generated a borderline response also identifies exactly what's wrong with it and produces a better version. That shouldn't work as well as it does. It implies the model has enough internalized alignment to *critique* a response even when its default generation didn't reflect that alignment. That asymmetry is strange and worth thinking about carefully.

**3. The sycophancy angle surprised me more than the harm-avoidance angle.** I came in focused on harmlessness. The paper also describes using CAI to reduce sycophancy — the tendency of RLHF-trained models to prefer agreeable responses even when they're wrong, because human raters reward agreement. CAI can hard-code honesty as a constitutional principle: "don't flatter the user, don't soften inconvenient truths when accuracy matters." For someone building a financial guidance tool, that failure mode is more dangerous than most explicit harms. A model that tells you what you want to hear about your portfolio is genuinely bad.

## My honest take

CAI is elegant. Replacing a human annotation bottleneck with model self-critique is one of those ideas that seems obvious in retrospect — the kind of thing that makes you wonder why it took as long as it did.

But the finite-constitution problem is real and shouldn't be papered over. The principles I defined cover the harms I anticipated. A novel attack vector — something the constitution's authors didn't think to include — has no catch mechanism. The model has no principle to critique against. Anthropic is explicit about this in the paper; CAI is one layer of a multi-layer defense system, not a complete solution. You still need red-teaming, evals, and human oversight at the frontier.

The thing that changed for me practically: I stopped thinking about system prompts as instructions and started thinking about them as a runtime constitution. When I write a system prompt now, I think about which internalized principles I'm asking the model to partially relax, and whether I've given it enough context to do that responsibly. The caveat-heavy behavior I was seeing in FinMentor wasn't my prompt failing — it was the model applying something like a constitutional check. Understanding that changes what I write in the system prompt and what I leave out.

## What's next

Up next: RLHF. I want to understand reward model training from the ground up — specifically where human preference data introduces systematic biases, and what the training dynamics look like when the reward model and the policy model update in lockstep. CAI is partly an answer to RLHF's annotation bottleneck. I want to understand the problem it's solving before I form strong opinions about whether the solution is sufficient.

Follow along: [https://github.com/saulolinares10/anthropic-alignment-notes](https://github.com/saulolinares10/anthropic-alignment-notes)
