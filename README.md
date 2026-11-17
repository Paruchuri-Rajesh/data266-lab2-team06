# DATA 266 Lab 2 - Pair_Programming_Team_06

| Part | Notebook | Runs on | Output |
|---|---|---|---|
| 1 Multimodal RAG | `notebooks/p1_rag.ipynb` | Kaggle T4 (final, internet ON) / Colab T4 (dev + golden eval) | `submission.csv`, `part1/results_table.csv` |
| 2 Diffusion LoRA | `notebooks/p2_finetune.ipynb` | Colab T4 | LoRA adapters + image dataset on HF, `part2/results.csv` |
| 3 Multi-agent system | `notebooks/p3_agents.ipynb` | Colab T4 | Gradio app, `part3/metrics.json`, traces |

All work fits one 16 GB GPU (T4). Models used are listed in `AI_USAGE.md`.

## How to run
Notebook sources live in `notebooks_src/*.py` (percent format) and are converted with
`python tools/py2nb.py notebooks_src/p1_rag.py notebooks/p1_rag.ipynb`.

**Part 1.** On Kaggle, attach the competition data, turn Internet on, pick a T4 GPU and Run All. The notebook reads
`QUESTIONS_PATH` (default `Questions.csv`) and writes `/kaggle/working/submission.csv` with no manual step.
On Colab, add the `KAGGLE_API_TOKEN` secret. The notebook then downloads the data itself and also runs the golden-set evaluation
(`part1/golden_set.csv`).
Env options: `VLM_ID`, `VLM_4BIT`, `USE_RERANKER`, `FIG_CITATION_STYLE`.

**Part 2.** On a Colab T4, set `REPO_URL`, `HF_USER` and the `HF_TOKEN` secret, then Run All. Training resumes from the latest checkpoint after a disconnect.

**Part 3.** On a Colab T4, set `HF_USER`/`FINAL_LORA` to the Part 2 run you chose, put test images in `part3/images/`, then Run All.
Set `RUN_APP=True` to start the Gradio demo.

Seeds: 266 everywhere (Part 2 generation uses seeds 0 and 1 as required).

## Second implementation by Siva Surya Chandran (`sivasurya_chandran/`)
Following the Lab 1 convention of one folder per member, `sivasurya_chandran/` holds a complete, self-contained second
implementation of all three parts (code, executed notebooks, results, human-evaluation tooling and files). It does not
change anything outside that folder. Its own `README.md`, `requirements.txt`, `AI_USAGE.md` and `CONTRIBUTIONS.md` describe how to run it
(`python -m p1_rag.run_competition`, or the notebooks in `sivasurya_chandran/notebooks/`, run from inside that folder).

| Part | What it is | Where |
|---|---|---|
| 1 | Hybrid dense + BM25 retrieval, table-first lookup, 4-bit Qwen2.5-14B answerer with an evidence check and a conflict pass; 4-bit Qwen2.5-VL-7B reads figures. Golden set 25 (baseline 23, final 22; Recall@5 0.913 to 1.000). Team-selected Kaggle file `submission_gh_v7.csv` came from this pipeline. | `sivasurya_chandran/p1_rag/`, `notebooks/p1_rag.ipynb` |
| 2 | LoRA rank 4 and rank 32 on Naruto BLIP captions (KID 0.123 base, 0.045 r4, 0.026 r32); blind comparison by three raters. | `sivasurya_chandran/p2_finetune/`, `docs/part2_model_card_and_final_run.md` |
| 3 | Five agents with routing rules and a deceptive-person-edit guardrail; 15/15 scenarios and 0/10 attacks in the final run (the first run, before fixes, was 10/15 and 2/10; both executed notebooks are kept in `notebooks/history/`). | `sivasurya_chandran/p3_agents/` |

All numbers there come from executed notebook outputs in that folder. The `sivasurya_chandran/` tree is the exact committed state of
https://github.com/sivasuryachandran/DATA266_LAB2 (see `links.txt` for its tag commits).
