# AI coding assistant usage

| Tool | Version | Used for |
|---|---|---|
| Claude Code (Anthropic) | Claude Opus 5.5 (`claude-opus-5-5`) | Generated the initial notebook scaffolding for Parts 1-3 (ingestion, retrieval, agent orchestration, training/eval code), repo layout and run guide. All code was reviewed, run and modified by the team. |

Not AI-generated: the golden set (Part 1), the prompts for Part 2, the scenarios and red-team cases (Part 3), all human evaluations, and the report text.

# Hosted models / APIs used (none are paid APIs; all open weights run on one T4)
- Part 1: Qwen/Qwen2.5-VL-7B-Instruct (4-bit nf4), BAAI/bge-small-en-v1.5, BAAI/bge-reranker-base (bonus)
- Part 2: stable-diffusion-v1-5/stable-diffusion-v1-5, openai/clip-vit-base-patch16, LPIPS (alex)
- Part 3: Qwen/Qwen2.5-VL-3B-Instruct, SD 1.5 + our Part 2 LoRA, timbrooks/instruct-pix2pix, openai/clip-vit-base-patch32
