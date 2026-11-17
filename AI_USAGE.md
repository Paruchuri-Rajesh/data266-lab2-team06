# AI coding assistant usage

| Tool | Version | Used for |
|---|---|---|
| Claude Code (Anthropic) | Claude Opus 5.5 (`claude-opus-5-5`) | Generated the initial notebook scaffolding for Parts 1-3 (ingestion, retrieval, agent orchestration, training/eval code), repo layout and run guide. All code was reviewed, run and modified by the team. |

Part 2: the 10 out-of-domain and compositional prompts in `part2/prompts.json` were drafted with Claude Code and reviewed by the team (the 10 in-domain prompts come from test-set captions).

Part 2: the interpretation text in the notebook markdown cells (dataset limitations, failure modes, final-run choice, license,
memorization) and the "Limitations" section of the Hugging Face model card were drafted with Claude Code from our own
results, grids and figures, then reviewed and edited by the team.

Part 1: the 25-question golden set (`part1/golden_set.csv`) was drafted with Claude Code from the three PDFs at the team's
request; every gold value was checked against the PDF text or the rendered chart, and it was committed before any tuning.

Part 3: the 15 scenarios (`part3/scenarios.json`) and 10 red-team cases (`part3/redteam.json`) were drafted with
Claude Code at the team's request, after looking at our image set, and were reviewed by the team and committed before
the evaluation run. The six scene photos were chosen by Claude Code from CC0 images on Wikimedia Commons (sources in
`part3/images/LICENSES.md`); the team member photo is our own.

Not AI-generated: all human evaluations (ratings and judgements) and the report text.

# Hosted models / APIs used (none are paid APIs; all open weights run on one T4)
- Part 1: Qwen/Qwen2.5-VL-7B-Instruct (4-bit nf4), BAAI/bge-small-en-v1.5, BAAI/bge-reranker-base (bonus)
- Part 2: stable-diffusion-v1-5/stable-diffusion-v1-5, openai/clip-vit-base-patch16, LPIPS (alex)
- Part 3: Qwen/Qwen2.5-VL-3B-Instruct, SD 1.5 + our Part 2 LoRA, timbrooks/instruct-pix2pix, openai/clip-vit-base-patch32
