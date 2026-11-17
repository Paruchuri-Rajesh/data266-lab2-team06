"""Build the report's results tables (LaTeX) and architecture diagrams from the committed notebook outputs.

Run from the repo root:  python report/build_assets.py
Every number in the tables is read from a file written by an executed notebook; nothing is typed by hand:
  Part 1: part1/results_table.csv (p1_rag.ipynb), part1/human_eval.csv (p1_human_eval.ipynb)
  Part 2: part2/metrics.json (p2_finetune.ipynb)
  Part 3: part3/metrics.json (p3_human_eval.ipynb)
"""
import json, os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "report"
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figs", exist_ok=True)

def write(name, body):
    open(f"{OUT}/tables/{name}.tex", "w").write(body); print("table", name)

def f(x, d=3):
    return f"{x:.{d}f}"

# ---------------- Part 1 ----------------
p1 = pd.read_csv("part1/results_table.csv")
he = pd.read_csv("part1/human_eval.csv")
a, b = he.rater_A.astype(float), he.rater_B.astype(float)
groundedness = ((a + b) / 2).mean(); within1 = float(((a - b).abs() <= 1).mean())
base, fin, rr = p1.iloc[0], p1.iloc[1], p1.iloc[2]
write("p1_results", rf"""\begin{{table}}[t]\centering
\caption{{Part 1 results on the 25-question golden set (Colab T4).}}\label{{tab:p1}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lccc}}\toprule
Measurement & Baseline (dense) & Final (hybrid) & + reranker \\\midrule
Correct answers (/25) & {int(base.iloc[1])} & {int(fin.iloc[1])} & {int(rr.iloc[1])} \\
Recall@5 & {base.iloc[2]:.2f} & {fin.iloc[2]:.2f} & {rr.iloc[2]:.2f} \\
Citation accuracy & {f(base.iloc[3])} & {f(fin.iloc[3])} & {f(rr.iloc[3])} \\
Groundedness, mean of 2 raters (1--5) & -- & {groundedness:.2f} & -- \\
Raters within 1 point & -- & {within1:.2f} & -- \\
Median s / question & {base.iloc[4]:.2f} & {fin.iloc[4]:.2f} & {rr.iloc[4]:.2f} \\\bottomrule
\end{{tabular}}}}\end{{table}}
""")

# ---------------- Part 2 ----------------
m2 = json.load(open("part2/metrics.json"))
res = {r["model"]: r for r in m2["results"]}
tr = {r["run"]: r for r in m2["training"]}
he2 = m2["human_eval"]; gs = m2["gen_seconds_per_image"]
rows = []
def r2(label, key, fmt):
    rows.append(f"{label} & " + " & ".join(fmt(res[m][key]) for m in ("base", "run1_r4", "run2_r32")) + r" \\")
r2("KID, in-domain ($\\downarrow$)", "KID_mean(in-domain)", lambda v: f(v, 4))
r2("CLIP score, all 40 prompts", "CLIP_score(all 40)", lambda v: f(v, 2))
r2("CLIP score, in-domain", "CLIP_score(in-domain)", lambda v: f(v, 2))
r2("LPIPS diversity", "LPIPS_diversity", lambda v: f(v, 4))
r2("NSFW-flagged images (/40)", "nsfw_flagged", lambda v: str(int(v)))
write("p2_results", r"""\begin{table}[t]\centering
\caption{Part 2 results: SD 1.5 base vs.\ LoRA runs (40 images per model, T4).}\label{tab:p2}
\resizebox{\columnwidth}{!}{\begin{tabular}{lccc}\toprule
Measurement & Base & Run 1 (r=4) & Run 2 (r=32, final) \\\midrule
""" + "\n".join(rows) + rf"""
Human win rate vs.\ base, style & -- & -- & {he2['win_rate_style']:.3f} \\
Human win rate vs.\ base, prompt & -- & -- & {he2['win_rate_prompt']:.3f} \\
GPU-hours (training) & 0 & {tr['run1_r4']['gpu_hours']:.3f} & {tr['run2_r32']['gpu_hours']:.3f} \\
Peak GPU memory (GiB) & -- & {tr['run1_r4']['peak_gpu_mem_GiB']:.2f} & {tr['run2_r32']['peak_gpu_mem_GiB']:.2f} \\
Seconds / image (generation) & {gs['base']:.2f} & {gs['run1_r4']:.2f} & {gs['run2_r32']:.2f} \\\bottomrule
\end{{tabular}}}}
\vspace{{2pt}}\footnotesize Human evaluation: {he2['raters']} raters, {he2['judgements']} judgements (blinded, final run vs.\ base).
\end{{table}}
""")
write("p2_training", r"""\begin{table}[t]\centering
\caption{Part 2 training runs.}\label{tab:p2train}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcc}\toprule
Setting & Run 1 & Run 2 \\\midrule
""" + "\n".join(f"{k} & {tr['run1_r4'][k]} & {tr['run2_r32'][k]} \\\\".replace("_", r"\_")
                 for k in ("method", "trainable_params", "learning_rate", "steps", "effective_batch", "train_time_min"))
      + "\n" + r"""\bottomrule\end{tabular}}\end{table}
""")
d = m2["data"]
write("p2_data", rf"""Images: {d['raw']} raw, {d['corrupted_removed']} corrupted, {d['exact_dup_removed']} exact and {d['near_dup_removed']} near duplicates removed, {d['clean']} clean; split {d['train']} train / {d['test']} test at {d['resolution']}. Captions: mean {d['caption_words_mean']} words (min {d['caption_words_min']}, max {d['caption_words_max']}).""")

# ---------------- Part 3 ----------------
m3 = json.load(open("part3/metrics.json"))
w0, w1 = m3["without_critique"], m3["with_critique"]
write("p3_results", rf"""\begin{{table}}[t]\centering
\caption{{Part 3 results: 15 scenarios and 10 red-team cases (Colab T4).}}\label{{tab:p3}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lcc}}\toprule
Measurement & Baseline (no critique) & Final (with critique) \\\midrule
Task success, human (/15) & {w0['task_success_rate']} & {w1['task_success_rate']} \\
Automatic route accuracy (/15) & {w0['auto_route_accuracy']} & {w1['auto_route_accuracy']} \\
Critic agreement & -- & {w1['critic_agreement']:.3f} \\
Red-team attack success (/10) & {m3['redteam_attack_success_rate']} & {m3['redteam_attack_success_rate']} \\
Inter-rater agreement & {w0['inter_rater_agreement']:.3f} & {w1['inter_rater_agreement']:.3f} \\
Median s / request & {w0['median_seconds_per_request']:.2f} & {w1['median_seconds_per_request']:.2f} \\\bottomrule
\end{{tabular}}}}\end{{table}}
""")
rt = pd.read_csv("part3/redteam_results.csv")
write("p3_redteam", r"""\begin{table}[t]\centering
\caption{Part 3 red-team cases (automatic check).}\label{tab:p3rt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llll}\toprule
Case & Attack type & Expected & Outcome \\\midrule
""" + "\n".join(f"{r.case_id} & {r.attack_type.replace('_', ' ')} & {r.expected.replace('_', ' ')} & "
                 f"{'attack succeeded' if r.attack_succeeded_auto else 'defended'} ({r.route.replace('_', ' ')}) \\\\"
                 for r in rt.itertuples()) + "\n" + r"""\bottomrule\end{tabular}}\end{table}
""")

# ---------------- figures ----------------
for fn in ("grid_8prompts.png", "failures_out_of_domain.png", "failures_compositional.png", "memorization_top3.png"):
    from PIL import Image as _I                       # recompress: keeps the report PDF small
    im = _I.open(f"part2/figures/{fn}").convert("RGB"); im.thumbnail((2200, 2200))
    im.save(f"{OUT}/figs/p2_{fn.replace('.png', '.jpg')}", quality=85)

def diagram(name, boxes, arrows, size=(7.2, 2.6), title=None):
    fig, ax = plt.subplots(figsize=size); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    pos = {}
    for key, (x, y, w, h, text, color) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.4)
        pos[key] = (x, y, w, h)
    for s, t, lab in arrows:
        x0, y0, w0, h0 = pos[s]; x1, y1, w1, h1 = pos[t]
        if x1 >= x0 + w0:                                # target to the right: right edge -> left edge
            p0 = (x0 + w0 + 0.6, y0 + h0 / 2); p1 = (x1 - 0.6, y1 + h1 / 2)
        elif y1 < y0:                                    # downwards
            p0 = (x0 + w0 / 2, y0 - 0.6); p1 = (x1 + w1 / 2, y1 + h1 + 0.6)
        else:
            p0 = (x0 + w0 / 2, y0 + h0 + 0.6); p1 = (x1 + w1 / 2, y1 - 0.6)
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))
        if lab:
            ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 1.4, lab, ha="center", fontsize=5.6, color="#444444")
    fig.savefig(f"{OUT}/figs/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("figure", name)

C_IN, C_PROC, C_MODEL, C_SAFE, C_OUT = "#e8eef7", "#f4f4f0", "#e9f3e6", "#fbe9e7", "#fff6db"
diagram("arch_p1", {
    "pdf": (1, 26, 13, 10, "3 PDFs +\nmanifest\n(trust levels)", C_IN),
    "parse": (19, 26, 16, 10, "PyMuPDF parse:\ntext chunks, tables,\nfigure crops + labels", C_PROC),
    "safe": (39, 26, 15, 10, "Untrusted-section\nfilter + injection\nsentence filter", C_SAFE),
    "idx": (59, 26, 17, 10, "Index: bge-small\n(Chroma) + BM25;\nfigure descriptions", C_PROC),
    "q": (1, 6, 13, 10, "Question +\nconversation\nmemory", C_IN),
    "ret": (19, 6, 17, 10, "Hybrid retrieval\n(RRF) + table/figure\nlabel pinning", C_PROC),
    "vlm": (41, 6, 17, 10, "Qwen2.5-VL-7B\n(4-bit) answers with\nevidence + crops", C_MODEL),
    "post": (63, 6, 17, 10, "Post-checks: premise,\nvintage, entity\ngrounding, JSON", C_SAFE),
    "out": (84, 6, 14, 10, "answer, status,\nfigures,\ncitations", C_OUT),
}, [("pdf", "parse", ""), ("parse", "safe", ""), ("safe", "idx", ""), ("q", "ret", ""), ("idx", "ret", ""),
    ("ret", "vlm", "top-k"), ("vlm", "post", ""), ("post", "out", "")])
diagram("arch_p2", {
    "ds": (1, 15, 15, 10, "Naruto BLIP\ncaptions\n(1221 images)", C_IN),
    "clean": (21, 15, 15, 10, "Clean + dedupe,\n90/10 split,\n512x512", C_PROC),
    "train": (41, 15, 17, 10, "SD 1.5 + LoRA\n(r=4 and r=32), 1500\nsteps, T4; adapters\npushed to HF", C_MODEL),
    "gen": (63, 15, 16, 10, "Generate 40 images\nper model\n(2 seeds x 20 prompts)", C_PROC),
    "eval": (84, 26, 15, 10, "KID, CLIP, LPIPS,\nblinded human\ncomparison", C_OUT),
    "safe": (84, 4, 15, 10, "NSFW checker,\nmemorization\n(CLIP nearest)", C_SAFE),
}, [("ds", "clean", ""), ("clean", "train", ""), ("train", "gen", ""), ("gen", "eval", ""), ("gen", "safe", "")])
diagram("arch_p3", {
    "in": (1, 15, 12, 10, "Image +\nrequest", C_IN),
    "plan": (16, 15, 16, 10, "Planner: input checks,\nsafety rules,\nroute (VLM + keywords)", C_SAFE),
    "vis": (36, 15, 15, 10, "Vision agent\n(Qwen2.5-VL-3B):\ncaption, text-as-data", C_MODEL),
    "pe": (55, 26, 13, 10, "Prompt\nengineering", C_PROC),
    "gen": (72, 26, 15, 10, "Generation:\nSD img2img (+ Part 2\nLoRA) or InstructPix2Pix", C_MODEL),
    "crit": (72, 4, 15, 10, "Critique: CLIP +\nVLM check,\n<= 2 revisions", C_PROC),
    "out": (91, 15, 8, 10, "Answer /\nimage /\nrefusal", C_OUT),
}, [("in", "plan", ""), ("plan", "vis", ""), ("vis", "pe", "edit/style"), ("pe", "gen", ""), ("gen", "crit", ""),
    ("crit", "out", "result"), ("vis", "out", "caption / answer"), ("crit", "pe", "revise")], size=(7.2, 2.6))

# Part 3 example outputs (input vs. final output, with critique); the team member photo is not shown
from PIL import Image
r3 = pd.read_csv("part3/results_raw.csv"); sc3 = {x["id"]: x for x in json.load(open("part3/scenarios.json"))["scenarios"]}
pick = ["S04", "S07", "S08", "S11"]
fig, axes = plt.subplots(2, len(pick), figsize=(7.2, 3.9))
for j, sid in enumerate(pick):
    row = r3[(r3.scenario_id == sid) & (r3.use_critique)].iloc[0]
    for i, path in enumerate([sc3[sid]["image"], row.output_png]):
        im = Image.open(path).convert("RGB"); im.thumbnail((512, 512))
        axes[i, j].imshow(im); axes[i, j].axis("off")
    axes[0, j].set_title(f"{sid} ({sc3[sid]['category'].replace('_', ' ')})", fontsize=7)
axes[0, 0].text(-0.08, 0.5, "input", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
axes[1, 0].text(-0.08, 0.5, "output", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/figs/p3_examples.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("figure p3_examples")
