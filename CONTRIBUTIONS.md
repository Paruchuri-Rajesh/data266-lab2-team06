# Contributions - Pair_Programming_Team_06

Members: **Rajesh Paruchuri** (member A, GitHub `Paruchuri-Rajesh`) and **Siva Surya Chandran** (member B, GitHub `sivasuryachandran`).
AI assistance is declared separately in `AI_USAGE.md`.

## Who did what

| Part | Rajesh Paruchuri (member A) | Siva (member B) |
|---|---|---|
| Part 1 - Multimodal RAG | Set up the repo and Kaggle/Colab secrets; ran the RAG notebook end to end on a Colab T4 (dense baseline, hybrid final, reranker bonus); reviewed the 25-question golden set against the PDFs before tuning; debugged retrieval, table pinning, false-premise and injection handling; ran the final Kaggle notebook (Version 2, T4) and checked `submission.csv` with the validator; human groundedness rater A. | Built and submitted an alternative Part 1 pipeline on Kaggle (file submissions `submission_gh_v3` 0.794 and `submission_gh_v7` 0.853, and a later `submission.csv` 0.858, the team's best and selected leaderboard submission); human groundedness rater B (independent scores); built a second, independent Part 1 implementation in `sivasurya_chandran/` (hybrid retrieval with table-first lookup, 4-bit Qwen2.5-14B answerer, evidence check and a conflict pass) with its own 25-question golden set and executed notebook, and a groundedness rating page (`sivasurya_chandran/human_eval_p1/`). |
| Part 2 - LoRA fine-tuning (lead: Rajesh) | Data cleaning and split; trained run 1 (LoRA r=4) and run 2 (LoRA r=32) on a Colab T4; generation, KID/CLIP/LPIPS evaluation, grids, memorization check; pushed both adapters, model cards and the generation dataset to Hugging Face; human rater. | Human rater for the blinded 25-pair style/prompt comparison (independent); ran the complete Part 2 pipeline (rank-4 and rank-32 LoRA runs, generation, KID/CLIP/LPIPS, memorization check) in `sivasurya_chandran/`, and built the blind rating page used by three raters for that run (`sivasurya_chandran/human_eval_p2/`). |
| Part 3 - Multi-agent image system (lead: <CHOOSE>) | Prepared the image set (CC0 photos + own photo with consent); ran the full evaluation on a Colab T4 (15 scenarios with/without critique, 10 red-team cases); pushed results; member A judgements. | Member B judgements (independent); the team adopted member B's judgements as `final_pass` when resolving disagreements; ran the complete Part 3 pipeline (15 scenarios with and without the critic, 10 red-team cases) in `sivasurya_chandran/`, including a judging page (`sivasurya_chandran/human_eval_p3/`); results are in `sivasurya_chandran/notebooks/p3_agents.ipynb`. |
| Report / demo | <FILL: sections written by Rajesh> | Appendix A (second implementation and results) of the shared report draft; the report generator `sivasurya_chandran/tools/build_report.py`; the Part 1, 2 and 3 human-evaluation pages |

A third, outside rater also scored the Part 2 blinded comparison (`part2/ratings_raw/`).

## Meeting recordings

Recordings of knowledge-transfer sessions and architecture / trade-off discussions (required by the team rules).

| Date | Topic | Link |
|---|---|---|
| 2026-11-14 | Lab 2 kickoff; Part 1 architecture (retrieval, answering rules) | <LINK> |
| <DATE> | Part 2 LoRA settings: rank 4 vs rank 32, choosing the final run | <LINK> |
| <DATE> | Part 3 agent flow, safety rules and red-team cases | <LINK> |
| <DATE> | Resolving human-evaluation disagreements; Kaggle submission choice | <LINK> |
