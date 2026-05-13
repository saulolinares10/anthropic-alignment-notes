I've been building on Claude for months and I realized I had no idea why it refused certain things. So I went back to the source.

I'm building FinMentor — a multi-agent financial advisor that runs on Claude. Four agents, connected to my IBKR account, daily driver. But I kept hitting this pattern: Claude would wrap perfectly good analysis in so many caveats it buried the answer. I rewrote my system prompts. Tried tighter instructions. Same behavior. That's when I stopped blaming my prompts.

That question sent me to Anthropic's Constitutional AI paper.

Here's the mechanism: instead of paying humans to label harmful outputs, you give the model a list of principles — the "constitution." The model generates a response, critiques its own output against one principle, rewrites it. That rewritten response becomes training data. No human in the loop. The same model plays both roles: generator and critic.

I built a simulation of this loop: three prompts designed to elicit manipulation-adjacent responses, five constitutional principles, two revision cycles each. I logged every intermediate state.

Here's the part that surprised me — and I don't think gets enough attention: the most valuable thing CAI does isn't harm avoidance. It's sycophancy reduction. Human RLHF raters prefer agreeable responses. CAI can hard-code honesty as a constitutional principle — the model critiques itself for being too agreeable, then revises. That's a harder problem than blocking explicit harm, and it matters far more when you're building something that gives financial guidance.

The honest limitation: the constitution is finite. Novel harms it didn't anticipate have no catch mechanism.

What do you think is harder to solve — preventing explicit harm, or building a model that's honest even when honesty is uncomfortable?

Full breakdown + working notebook: [dev.to link — replace before publishing]

#AIAlignment #Anthropic #LLM #BuildingInPublic #MachineLearning #Python #Claude
