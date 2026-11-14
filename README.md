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
