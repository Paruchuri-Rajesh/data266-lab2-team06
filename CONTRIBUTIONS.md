# Contributions - Pair_Programming_Team_06

Members: **Rajesh Paruchuri** (member A, GitHub `Paruchuri-Rajesh`) and **Siva Surya Chandran** (member B, GitHub `sivasuryachandran`).
Team lead: **Siva Surya Chandran**.
AI assistance is declared separately in `AI_USAGE.md`.

## Who did what

| Part | Rajesh Paruchuri (member A) | Siva Surya Chandran (member B, team lead) |
|---|---|---|
| Part 1 - Multimodal RAG | Set up the repo and Kaggle/Colab secrets; ran the RAG notebook end to end on a Colab T4 (dense baseline, hybrid final, reranker bonus); reviewed the 25-question golden set against the PDFs before tuning; debugged retrieval, table pinning, false-premise and injection handling; ran the final Kaggle notebook (Version 2, T4) and checked `submission.csv` with the validator; human groundedness rater A. | Built and submitted an alternative Part 1 pipeline on Kaggle (file submissions `submission_gh_v3` 0.794 and `submission_gh_v7` 0.853, and a later `submission.csv` 0.858, the team's best and selected leaderboard submission); human groundedness rater B (independent scores); built a second, independent Part 1 implementation in `sivasurya_chandran/` (hybrid retrieval with table-first lookup, 4-bit Qwen2.5-14B answerer, evidence check and a conflict pass) with its own 25-question golden set and executed notebook, and a groundedness rating page (`sivasurya_chandran/human_eval_p1/`); final Part 1 Kaggle notebook `sivasuryachandran/data266-team06-p1-final` (Version 4, 2x T4, 63.6 min end to end, validator: valid 40 rows). |
| Part 2 - LoRA fine-tuning (lead: Rajesh) | Data cleaning and split; trained run 1 (LoRA r=4) and run 2 (LoRA r=32) on a Colab T4; generation, KID/CLIP/LPIPS evaluation, grids, memorization check; pushed both adapters, model cards and the generation dataset to Hugging Face; human rater. | Human rater for the blinded 25-pair style/prompt comparison (independent); ran the complete Part 2 pipeline (rank-4 and rank-32 LoRA runs, generation, KID/CLIP/LPIPS, memorization check) in `sivasurya_chandran/`, and built the blind rating page used by three raters for that run (`sivasurya_chandran/human_eval_p2/`). |
| Part 3 - Multi-agent image system (lead: Siva Surya Chandran) | Prepared the image set (CC0 photos + own photo with consent); ran the full evaluation on a Colab T4 (15 scenarios with/without critique, 10 red-team cases); pushed results; member A judgements. | Member B judgements (independent); the team adopted member B's judgements as `final_pass` when resolving disagreements; ran the complete Part 3 pipeline (15 scenarios with and without the critic, 10 red-team cases) in `sivasurya_chandran/`, including a judging page (`sivasurya_chandran/human_eval_p3/`); results are in `sivasurya_chandran/notebooks/p3_agents.ipynb`. |
| Report / demo | First half of the report (the PDF is exported from the shared Google Doc): abstract, I Introduction, II Part 1 (Multimodal RAG) and III Part 2 (LoRA fine-tuning) | Second half of the report: IV Part 3 (multi-agent image system), V Conclusion, the Contributions and AI usage section, and Appendix A (second implementation and results); the report generator `sivasurya_chandran/tools/build_report.py`; the Part 1, 2 and 3 human-evaluation pages |

A third, outside rater also scored the Part 2 blinded comparison (`part2/ratings_raw/`).

## Code links

Recordings folder: https://drive.google.com/drive/folders/1euWx7fMLZlHuzdXfgy8KJZxX_kdIOhW-

| What | Link |
|---|---|
| Team repo (main implementation, Rajesh) | https://github.com/Paruchuri-Rajesh/data266-lab2-team06 - tags `p1-v1.0` (4dd5fc7), `p2-v1.0` (f54a892), `p3-v1.0` (d11880a) |
| Second implementation (Siva, report Appendix A) | https://github.com/sivasuryachandran/DATA266_LAB2 - tags `p1-v1.0` (2698425), `p2-v1.0` (74dde8d), `p3-v1.0` (74dde8d) |
| Pull request adding Siva's work to the team repo | https://github.com/Paruchuri-Rajesh/data266-lab2-team06/pull/1 |

## Team meetings

Team discussions (knowledge transfer, architecture and trade-offs) were held on SJSU Zoom. The links below are the Zoom
meeting links we used.

| Meeting | Zoom meeting link |
|---|---|
| Meeting 1 | https://sjsu.zoom.us/j/89910572692?pwd=3LMjaV70mf6GcqgFBnV4oHHQNzB2Oa.1 |
| Meeting 2 | https://sjsu.zoom.us/j/81885574080?pwd=1ZpaNtuOSPAxompNUIVNaE6buV6SgM.1 |
| Meeting 3 | https://sjsu.zoom.us/j/88046496026?pwd=UWO0b4x4MdiziUpWCdUbGi05U8xH74.1 |
