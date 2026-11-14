# %% [markdown]
# # DATA 266 Lab 2 — Part 2: Fine-Tuning Stable Diffusion 1.5 (LoRA) on Naruto BLIP Captions
#
# **Hardware:** one 16 GB GPU (Colab **T4** / Kaggle T4). In Colab: *Runtime → Change runtime type → T4 GPU*.
#
# **Runs:** Run 1 = LoRA rank 4, Run 2 = LoRA rank 32 (same data, same step budget, same seed).
#
# **How to use this notebook**
# 1. Run Cell 1 (required run info) and the Setup cells.
# 2. After any Colab disconnect: re-run Cell 1 → Setup → Config, then continue from the section you were in.
#    Every section reloads what it needs from disk, and training resumes from the last checkpoint.
# 3. Before submitting: *Runtime → Restart and run all* (or re-run top to bottom) so all outputs are saved.

# %%
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})

# %% [markdown]
# ## Setup: Drive, repo clone, Hugging Face login
# Fill in the three placeholders below. Store your Hugging Face **write** token as a Colab secret named `HF_TOKEN`
# (key icon in the left sidebar) — never paste it into the notebook.

# %%
import os
REPO_URL = "https://github.com/Paruchuri-Rajesh/data266-lab2-team06.git"   # TODO: your private repo
HF_USER  = ""     # leave empty: filled automatically from your HF_TOKEN after login
TEAM     = "06"
USE_DRIVE = True   # keep training outputs on Google Drive so a disconnect does not lose them

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

REPO_DIR = "/content/data266-lab2-team" + TEAM if IN_COLAB else os.getcwd()
if IN_COLAB and not os.path.isdir(REPO_DIR):
    # Private repo: use a GitHub token stored as Colab secret GH_TOKEN
    try:
        from google.colab import userdata
        gh = userdata.get("GH_TOKEN")
        url = REPO_URL.replace("https://", f"https://{gh}@") if gh else REPO_URL
    except Exception:
        url = REPO_URL
    subprocess.run(["git", "clone", url, REPO_DIR], check=True)
if os.path.basename(os.getcwd()) != "notebooks" and os.path.isdir(REPO_DIR):
    os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# Re-print run info now that we are inside the repo (commit is now known)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})

# %%
!pip install -q diffusers==0.31.0 "transformers>=4.44,<4.47" "accelerate>=0.34,<1.2" "peft>=0.13,<0.14" "datasets>=2.21,<3.2" "torchmetrics[image]==1.4.2" lpips==0.1.4 imagehash==4.3.1 "huggingface_hub>=0.25,<0.27" safetensors tensorboard
!wget -q -nc https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/text_to_image/train_text_to_image_lora.py -O train_text_to_image_lora.py
!ls -la train_text_to_image_lora.py

# %%
from huggingface_hub import login
try:
    from google.colab import userdata
    login(token=userdata.get("HF_TOKEN"))
except Exception:
    tok = os.environ.get("HF_TOKEN")
    login(token=tok) if tok else login()   # interactive fallback

from huggingface_hub import whoami
if not HF_USER:
    HF_USER = whoami()["name"]
print("HF_USER =", HF_USER)

import diffusers, transformers, accelerate, peft, datasets, torchmetrics
print({"diffusers": diffusers.__version__, "transformers": transformers.__version__,
       "accelerate": accelerate.__version__, "peft": peft.__version__,
       "datasets": datasets.__version__, "torchmetrics": torchmetrics.__version__})

# %% [markdown]
# ## Config (re-run after every restart)

# %%
import json, time, glob, hashlib, io, shutil, threading, gc
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datasets import Image as HFImage

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_ID = "lambdalabs/naruto-blip-captions"
RES = 512

# Large files (dataset, checkpoints, images) live in WORK (Drive if mounted); small results go to the repo (part2/)
WORK = Path("/content/drive/MyDrive/data266_lab2/p2") if (IN_COLAB and USE_DRIVE) else Path("/content/p2_work" if IN_COLAB else "p2_work")
DATA_DIR, RUNS_DIR, GEN_DIR = WORK / "data", WORK / "runs", WORK / "generations"
OUT = Path("part2"); OUT.mkdir(exist_ok=True)
for d in (DATA_DIR, RUNS_DIR, GEN_DIR): d.mkdir(parents=True, exist_ok=True)

# Training hyper-parameters — identical for both runs except the LoRA rank
TRAIN_CFG = dict(max_train_steps=1500, train_batch_size=1, gradient_accumulation_steps=4,
                 learning_rate=1e-4, lr_scheduler="cosine", lr_warmup_steps=0,
                 checkpointing_steps=500, seed=SEED, resolution=RES, mixed_precision="fp16")
RUNS = {"run1_r4": 4, "run2_r32": 32}
FINAL_RUN = "run2_r32"   # TODO: set to the run you choose as final after looking at Section 2.4 results

# Sampler settings — identical for every model
GEN_CFG = dict(num_inference_steps=30, guidance_scale=7.5, height=RES, width=RES)
GEN_SEEDS = [0, 1]
HF_LORA_REPOS = {name: f"{HF_USER}/data266-naruto-lora-r{r}" for name, r in RUNS.items()}
HF_GEN_DATASET = f"{HF_USER}/data266-p2-generations"
print(WORK, HF_LORA_REPOS, HF_GEN_DATASET)

def load_prompts():
    P = json.load(open(OUT / "prompts.json"))
    return P, [(cat, p) for cat in ("in_domain", "out_of_domain", "compositional") for p in P[cat]]
P, PROMPT_LIST = load_prompts()
MODELS = {"base": None, "run1_r4": RUNS_DIR / "run1_r4", "run2_r32": RUNS_DIR / "run2_r32"}

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## 2.1 Data Preparation
# Load the dataset, remove corrupted images and duplicates (exact MD5 + identical perceptual hash),
# split 90/10 with seed 266, center-crop/resize to 512×512, and save as an `imagefolder` with `metadata.jsonl`.

# %%
import imagehash
from datasets import load_dataset

raw = load_dataset(DATASET_ID, split="train").cast_column("image", HFImage(decode=False))
print("raw rows:", len(raw))

def center_crop_resize(img, size=RES):
    w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.BICUBIC)

kept, n_corrupt, n_md5_dup, n_phash_dup = [], 0, 0, 0
seen_md5, seen_phash = set(), set()
for i, row in enumerate(raw):
    b = row["image"].get("bytes")
    if b is None and row["image"].get("path"):
        b = open(row["image"]["path"], "rb").read()
    cap = (row.get("text") or "").strip()
    try:
        if not b or len(b) == 0 or not cap:
            raise ValueError("empty image bytes or caption")
        Image.open(io.BytesIO(b)).verify()                 # structural check
        img = Image.open(io.BytesIO(b)).convert("RGB")     # full decode
        if min(img.size) < 64:
            raise ValueError("too small")
    except Exception:
        n_corrupt += 1; continue
    md5 = hashlib.md5(b).hexdigest()
    if md5 in seen_md5:
        n_md5_dup += 1; continue
    ph = str(imagehash.phash(img))
    if ph in seen_phash:                                   # near-duplicate (phash distance 0)
        n_phash_dup += 1; continue
    seen_md5.add(md5); seen_phash.add(ph)
    kept.append((img, cap))

print({"raw": len(raw), "corrupted_removed": n_corrupt, "exact_duplicates_removed": n_md5_dup,
       "near_duplicates_removed(phash=0)": n_phash_dup, "kept": len(kept)})

# %%
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(kept))
n_test = int(round(0.10 * len(kept)))
test_idx, train_idx = sorted(perm[:n_test]), sorted(perm[n_test:])

def save_split(indices, folder):
    folder = Path(folder)
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with open(folder / "metadata.jsonl", "w") as f:
        for j, k in enumerate(indices):
            img, cap = kept[k]
            fn = f"{j:05d}.png"
            center_crop_resize(img).save(folder / fn)
            f.write(json.dumps({"file_name": fn, "text": cap}) + "\n")

save_split(train_idx, DATA_DIR / "train")
save_split(test_idx, DATA_DIR / "test")
test_caps = [kept[k][1] for k in test_idx]

cap_len = pd.Series([len(c.split()) for _, c in kept])
data_summary = {"raw": len(raw), "corrupted_removed": n_corrupt, "exact_dup_removed": n_md5_dup,
                "near_dup_removed": n_phash_dup, "clean": len(kept), "train": len(train_idx), "test": len(test_idx),
                "caption_words_mean": round(float(cap_len.mean()), 2), "caption_words_min": int(cap_len.min()),
                "caption_words_max": int(cap_len.max()), "resolution": f"{RES}x{RES}"}
json.dump(data_summary, open(OUT / "data_summary.json", "w"), indent=2)
print(pd.Series(data_summary))
print("\nExample captions:"); [print(" -", c) for c in test_caps[:5]]
del kept; gc.collect()

# %%
# Quick visual check of the training data
tr = [json.loads(l) for l in open(DATA_DIR / "train" / "metadata.jsonl")]
fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
for ax, m in zip(axes, tr[:6]):
    ax.imshow(Image.open(DATA_DIR / "train" / m["file_name"])); ax.axis("off")
    ax.set_title(m["text"][:40], fontsize=8)
plt.show()

# %% [markdown]
# **Dataset description (write in your own words):**
# - TODO: what the dataset contains (use the numbers printed above).
# - TODO Limitation / bias 1: …
# - TODO Limitation / bias 2: …

# %% [markdown]
# ## 2.3a Fix the 20 evaluation prompts BEFORE training
# This cell fills the 10 in-domain prompts from test-set captions (seed 266) if they are still empty.
# **Replace the out-of-domain and compositional examples in `part2/prompts.json` with your own, then `git commit` and
# `git push` the file before training ends** (the Git timestamp is checked).

# %%
PROMPTS_PATH = OUT / "prompts.json"
P = json.load(open(PROMPTS_PATH))
if not P["in_domain"]:
    test_meta = [json.loads(l) for l in open(DATA_DIR / "test" / "metadata.jsonl")]
    pick = np.random.default_rng(SEED).choice(len(test_meta), 10, replace=False)
    P["in_domain"] = [test_meta[i]["text"] for i in sorted(pick)]
    json.dump(P, open(PROMPTS_PATH, "w"), indent=2)
    print("in_domain prompts written — now commit part2/prompts.json")
assert len(P["in_domain"]) == 10 and len(P["out_of_domain"]) == 5 and len(P["compositional"]) == 5
for cat in ("in_domain", "out_of_domain", "compositional"):
    print(f"\n{cat}:"); [print("  -", p) for p in P[cat]]
print("\nLast commit touching prompts.json:")
print(subprocess.run(["git", "log", "-1", "--format=%h %ci", "--", str(PROMPTS_PATH)], capture_output=True, text=True).stdout or "NOT COMMITTED YET")

# %% [markdown]
# ## 2.2 Fine-Tuning: two LoRA runs (rank 4 vs rank 32)
# Uses the official diffusers example `train_text_to_image_lora.py` (v0.31.0).
# Peak GPU memory is measured by polling `nvidia-smi` in a background thread; wall time is accumulated across
# resumed sessions in `timing.json`, so a Colab disconnect does not lose the measurement.
# Each run takes roughly 45–70 min on a T4.

# %%
class GpuPeakMonitor:
    """Polls nvidia-smi every second and records peak used memory (MiB) of GPU 0."""
    def __init__(self): self.peak, self._stop = 0, threading.Event()
    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                     capture_output=True, text=True).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(1)
    def __enter__(self): self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self._stop.set(); self.t.join()

def train_run(name, rank):
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_f = out_dir / "timing.json"
    timing = json.load(open(timing_f)) if timing_f.exists() else {"seconds": 0.0, "peak_mib": 0, "sessions": 0}
    if (out_dir / "pytorch_lora_weights.safetensors").exists():
        print(f"[{name}] already finished — skipping"); return timing
    c = TRAIN_CFG
    cmd = ["accelerate", "launch", "--num_processes", "1", "--num_machines", "1", "--mixed_precision", c["mixed_precision"],
           "--dynamo_backend", "no", "train_text_to_image_lora.py",
           "--pretrained_model_name_or_path", BASE_MODEL,
           "--train_data_dir", str(DATA_DIR / "train"), "--caption_column", "text", "--image_column", "image",
           "--resolution", str(c["resolution"]), "--center_crop", "--random_flip",
           "--train_batch_size", str(c["train_batch_size"]), "--gradient_accumulation_steps", str(c["gradient_accumulation_steps"]),
           "--gradient_checkpointing", "--max_train_steps", str(c["max_train_steps"]),
           "--learning_rate", str(c["learning_rate"]), "--lr_scheduler", c["lr_scheduler"], "--lr_warmup_steps", str(c["lr_warmup_steps"]),
           "--max_grad_norm", "1", "--rank", str(rank), "--seed", str(c["seed"]), "--mixed_precision", c["mixed_precision"],
           "--checkpointing_steps", str(c["checkpointing_steps"]), "--checkpoints_total_limit", "2",
           "--resume_from_checkpoint", "latest", "--dataloader_num_workers", "2",
           "--output_dir", str(out_dir), "--report_to", "tensorboard"]
    print(" ".join(cmd))
    log = open(out_dir / "train.log", "a")
    t0 = time.time()
    with GpuPeakMonitor() as mon:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        n = 0
        for line in proc.stdout:
            log.write(line); n += 1
            if "Steps:" not in line or n % 200 == 0:   # throttle tqdm spam
                print(line.rstrip()[-200:])
        proc.wait()
    timing["seconds"] += time.time() - t0
    timing["peak_mib"] = max(timing["peak_mib"], mon.peak)
    timing["sessions"] += 1
    json.dump(timing, open(timing_f, "w"), indent=2)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"[{name}] training failed (see {out_dir/'train.log'})")
    return timing

# %%
free_gpu()
timing_r1 = train_run("run1_r4", RUNS["run1_r4"])
print(timing_r1)

# %%
free_gpu()
timing_r2 = train_run("run2_r32", RUNS["run2_r32"])
print(timing_r2)

# %%
from safetensors.torch import load_file

def count_lora_params(name):
    sd = load_file(str(RUNS_DIR / name / "pytorch_lora_weights.safetensors"))
    return int(sum(t.numel() for t in sd.values()))

rows = []
for name, rank in RUNS.items():
    t = json.load(open(RUNS_DIR / name / "timing.json"))
    rows.append({"run": name, "method": f"LoRA r={rank}", "trainable_params": count_lora_params(name),
                 "learning_rate": TRAIN_CFG["learning_rate"], "steps": TRAIN_CFG["max_train_steps"],
                 "batch_size(per_step)": TRAIN_CFG["train_batch_size"],
                 "grad_accum": TRAIN_CFG["gradient_accumulation_steps"],
                 "effective_batch": TRAIN_CFG["train_batch_size"] * TRAIN_CFG["gradient_accumulation_steps"],
                 "peak_gpu_mem_GiB": round(t["peak_mib"] / 1024, 2),
                 "train_time_min": round(t["seconds"] / 60, 1), "gpu_hours": round(t["seconds"] / 3600, 3),
                 "sessions": t["sessions"]})
train_df = pd.DataFrame(rows)
train_df.to_csv(OUT / "train_runs.csv", index=False)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
train_df

# %% [markdown]
# ## 2.3 Image Generation — 20 prompts × 2 seeds × 3 models (base, run 1, run 2)
# Same scheduler (DPM-Solver++ multistep), 30 steps, guidance 7.5, 512×512. The SD safety checker stays **on**;
# its NSFW flag is recorded for every image. ~10–15 min on a T4.

# %%
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

P, PROMPT_LIST = load_prompts()
assert len(PROMPT_LIST) == 20, "fill part2/prompts.json first (Section 2.3a)"

def make_pipe(lora_dir=None):
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)  # safety checker ON
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir), weight_name="pytorch_lora_weights.safetensors")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")

meta_rows, gen_times = [], {}
for mname, ldir in MODELS.items():
    (GEN_DIR / mname).mkdir(parents=True, exist_ok=True)
    pipe = make_pipe(ldir); t0 = time.time()
    for pi, (cat, prompt) in enumerate(PROMPT_LIST):
        for seed in GEN_SEEDS:
            fn = GEN_DIR / mname / f"p{pi:02d}_s{seed}.png"
            g = torch.Generator("cuda").manual_seed(seed)
            out = pipe(prompt, generator=g, **GEN_CFG)
            out.images[0].save(fn)
            nsfw = bool(out.nsfw_content_detected[0]) if out.nsfw_content_detected is not None else False
            meta_rows.append({"file": str(fn.relative_to(GEN_DIR)), "model": mname, "prompt_idx": pi, "category": cat,
                              "prompt": prompt, "seed": seed, "nsfw_flag": nsfw, **GEN_CFG, "scheduler": "DPMSolverMultistep"})
    gen_times[mname] = (time.time() - t0) / (len(PROMPT_LIST) * len(GEN_SEEDS))
    del pipe; free_gpu()
    print(f"{mname}: done, {gen_times[mname]:.2f} s/image")

gen_meta = pd.DataFrame(meta_rows)
gen_meta.to_csv(GEN_DIR / "metadata.csv", index=False)
gen_meta.drop(columns=["height", "width"]).to_csv(OUT / "generation_metadata.csv", index=False)
json.dump(gen_times, open(OUT / "gen_seconds_per_image.json", "w"), indent=2)
print(gen_meta.groupby("model").size())

# %%
# Side-by-side grids (base | run1 | run2) for 8 prompts: 4 in-domain, 2 out-of-domain, 2 compositional (seed 0)
gen_meta = pd.read_csv(GEN_DIR / "metadata.csv")
GRID_PROMPTS = [0, 1, 2, 3, 10, 11, 15, 16]
(GEN_DIR / "grids").mkdir(exist_ok=True)
fig, axes = plt.subplots(len(GRID_PROMPTS), 3, figsize=(10, 3.4 * len(GRID_PROMPTS)))
for r, pi in enumerate(GRID_PROMPTS):
    for c, m in enumerate(MODELS):
        ax = axes[r, c]; ax.imshow(Image.open(GEN_DIR / m / f"p{pi:02d}_s0.png")); ax.axis("off")
        if r == 0: ax.set_title(m, fontsize=12)
    axes[r, 0].text(-20, 256, f"[{PROMPT_LIST[pi][0]}] {PROMPT_LIST[pi][1][:60]}", rotation=90, va="center", ha="right", fontsize=7)
plt.tight_layout(); plt.savefig(GEN_DIR / "grids" / "grid_8prompts.png", dpi=110); plt.show()

# %%
# Upload all 120 generated images (+ prompt, seed, model metadata) to a Hugging Face dataset
from datasets import Dataset
gen_meta = pd.read_csv(GEN_DIR / "metadata.csv")
ds = Dataset.from_pandas(gen_meta.assign(image=[str(GEN_DIR / f) for f in gen_meta["file"]])).cast_column("image", HFImage())
ds.push_to_hub(HF_GEN_DATASET, private=False)
print("pushed:", f"https://huggingface.co/datasets/{HF_GEN_DATASET}")

# %% [markdown]
# ## 2.4 Evaluation
# - **Primary:** KID (in-domain generations vs test images, lower is better; noisy with only 20 images) and CLIP score
#   (100 × cosine similarity between prompt and image, `openai/clip-vit-base-patch16`).
# - **Secondary:** diversity = mean LPIPS (AlexNet) between the seed-0 and seed-1 images of each prompt.
# - **Efficiency:** GPU-hours per training run (with peak GPU memory), from the table in 2.2.

# %%
from torchmetrics.image.kid import KernelInceptionDistance
from transformers import CLIPModel, CLIPProcessor
import lpips
import torchvision.transforms.functional as TF

dev = "cuda"
gen_meta = pd.read_csv(GEN_DIR / "metadata.csv")
test_meta = [json.loads(l) for l in open(DATA_DIR / "test" / "metadata.jsonl")]
to_t = lambda p: TF.to_tensor(Image.open(p).convert("RGB").resize((RES, RES)))   # float [0,1], CHW

real = torch.stack([to_t(DATA_DIR / "test" / m["file_name"]) for m in test_meta])

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

@torch.no_grad()
def clip_score(img_path, text):
    inp = clip_proc(text=[text], images=Image.open(img_path).convert("RGB"), return_tensors="pt",
                    padding=True, truncation=True, max_length=77).to(dev)
    o = clip_model(**inp)
    i, t = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True), o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
    return float(100 * torch.clamp((i * t).sum(), min=0))

lp = lpips.LPIPS(net="alex").to(dev)
results = []
for mname in MODELS:
    g = gen_meta[gen_meta.model == mname]
    # KID: in-domain generations (10 prompts x 2 seeds = 20 images) vs test images
    fake = torch.stack([to_t(GEN_DIR / f) for f in g[g.category == "in_domain"]["file"]])
    torch.manual_seed(SEED)
    kid = KernelInceptionDistance(subset_size=min(10, len(fake), len(real)), subsets=100, normalize=True).to(dev)
    for i in range(0, len(real), 32): kid.update(real[i:i + 32].to(dev), real=True)
    kid.update(fake.to(dev), real=False)
    kid_mean, kid_std = [float(x) for x in kid.compute()]
    del kid; free_gpu()
    # CLIP score: every prompt-image pair
    cs = [clip_score(GEN_DIR / r.file, r.prompt) for r in g.itertuples()]
    cs_in = [clip_score(GEN_DIR / r.file, r.prompt) for r in g[g.category == "in_domain"].itertuples()]
    # LPIPS diversity between seed 0 and seed 1 of each prompt
    lps = []
    for pi in sorted(g.prompt_idx.unique()):
        a = to_t(GEN_DIR / mname / f"p{pi:02d}_s0.png").unsqueeze(0).to(dev) * 2 - 1
        b = to_t(GEN_DIR / mname / f"p{pi:02d}_s1.png").unsqueeze(0).to(dev) * 2 - 1
        with torch.no_grad(): lps.append(float(lp(a, b)))
    row = {"model": mname, "KID_mean(in-domain)": kid_mean, "KID_std": kid_std,
           "CLIP_score(all 40)": float(np.mean(cs)), "CLIP_score(in-domain)": float(np.mean(cs_in)),
           "LPIPS_diversity": float(np.mean(lps)), "nsfw_flagged": int(g.nsfw_flag.sum())}
    if mname in RUNS:
        tr = train_df.set_index("run").loc[mname] if "train_df" in globals() else pd.read_csv(OUT / "train_runs.csv").set_index("run").loc[mname]
        row.update({"train_gpu_hours": float(tr["gpu_hours"]), "peak_gpu_mem_GiB": float(tr["peak_gpu_mem_GiB"])})
    else:
        row.update({"train_gpu_hours": 0.0, "peak_gpu_mem_GiB": float("nan")})
    results.append(row); print(row)

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "results.csv", index=False)
res_df.round(4)

# %% [markdown]
# ### Human evaluation — 25 blinded pairs (final run vs base model)
# The cell below builds `WORK/human_eval/` with 25 pairs (left/right randomised with seed 266), an `index.html` page
# for raters, a ratings template, and a **separate key file**. Give raters only the HTML page / images (not the key).
# At least 3 raters: both team members + one person outside the team. Each rater picks L or R for
# (a) domain style and (b) prompt match. Fill `part2/human_ratings.csv` with columns
# `rater,pair_id,style_choice,prompt_choice` and commit it.

# %%
HE = WORK / "human_eval"; (HE / "img").mkdir(parents=True, exist_ok=True)
cands = [(pi, s) for pi in range(len(PROMPT_LIST)) for s in GEN_SEEDS]
rng = np.random.default_rng(SEED)
chosen = [cands[i] for i in sorted(rng.choice(len(cands), 25, replace=False))]
key_rows, html = [], ["<html><body style='font-family:sans-serif'><h2>DATA 266 P2 — blinded comparison</h2>",
                      "<p>For each pair choose L or R for (a) Naruto domain style and (b) match to the prompt.</p>"]
for k, (pi, s) in enumerate(chosen, 1):
    final_left = bool(rng.integers(0, 2))
    L, R = (FINAL_RUN, "base") if final_left else ("base", FINAL_RUN)
    shutil.copy(GEN_DIR / L / f"p{pi:02d}_s{s}.png", HE / "img" / f"pair{k:02d}_L.png")
    shutil.copy(GEN_DIR / R / f"p{pi:02d}_s{s}.png", HE / "img" / f"pair{k:02d}_R.png")
    key_rows.append({"pair_id": k, "prompt_idx": pi, "seed": s, "left_model": L, "right_model": R})
    html.append(f"<h3>Pair {k}: {PROMPT_LIST[pi][1]}</h3><img src='img/pair{k:02d}_L.png' width=384> "
                f"<img src='img/pair{k:02d}_R.png' width=384><p>L | R</p><hr>")
open(HE / "index.html", "w").write("\n".join(html + ["</body></html>"]))
pd.DataFrame(key_rows).to_csv(OUT / "human_eval_key.csv", index=False)       # keep away from raters
pd.DataFrame([{"rater": r, "pair_id": k, "style_choice": "", "prompt_choice": ""}
              for r in ("member1", "member2", "outside_rater") for k in range(1, 26)]).to_csv(OUT / "human_ratings_template.csv", index=False)
print("Rater package:", HE, "| zip it and share index.html + img/ with raters")
shutil.make_archive(str(WORK / "human_eval_package"), "zip", HE)

# %%
# Win rate of the final run vs the base model (only runs once the real ratings file exists)
ratings_f = OUT / "human_ratings.csv"
human = None
if not ratings_f.exists():
    print("part2/human_ratings.csv not found yet — collect ratings from >=3 raters (copy human_ratings_template.csv), "
          "commit it, then re-run this cell. No numbers are reported until then.")
else:
    rt = pd.read_csv(ratings_f).merge(pd.read_csv(OUT / "human_eval_key.csv"), on="pair_id")
    for col in ("style_choice", "prompt_choice"):
        rt[col + "_final_wins"] = np.where(rt[col].str.upper() == "L", rt.left_model, rt.right_model) == FINAL_RUN
    human = {"raters": int(rt.rater.nunique()), "judgements": len(rt),
             "win_rate_style": float(rt["style_choice_final_wins"].mean()),
             "win_rate_prompt": float(rt["prompt_choice_final_wins"].mean())}
    print(human)
    print(rt.groupby("rater")[["style_choice_final_wins", "prompt_choice_final_wins"]].mean())

# %% [markdown]
# ### Failure modes
# The grids below show the out-of-domain and compositional prompts (seed 0 and 1) for all three models.

# %%
for cat in ("out_of_domain", "compositional"):
    idxs = [i for i, (c, _) in enumerate(PROMPT_LIST) if c == cat]
    fig, axes = plt.subplots(len(idxs), 6, figsize=(18, 3.2 * len(idxs)))
    for r, pi in enumerate(idxs):
        for c, (m, s) in enumerate([(m, s) for m in MODELS for s in GEN_SEEDS]):
            axes[r, c].imshow(Image.open(GEN_DIR / m / f"p{pi:02d}_s{s}.png")); axes[r, c].axis("off")
            if r == 0: axes[r, c].set_title(f"{m} s{s}", fontsize=9)
        axes[r, 0].set_ylabel(PROMPT_LIST[pi][1][:40], fontsize=7)
        print(f"[{cat}] p{pi:02d}: {PROMPT_LIST[pi][1]}")
    plt.suptitle(cat); plt.tight_layout(); plt.savefig(GEN_DIR / "grids" / f"failures_{cat}.png", dpi=90); plt.show()

# %% [markdown]
# **Failure mode 1 (your words, cite prompt numbers above):** TODO
#
# **Failure mode 2:** TODO
#
# **Which run we choose as final and why:** TODO (refer to `results.csv`, the grids and the human win rate)

# %% [markdown]
# ## Safety (C4)
# **License:** TODO in your own words — the Naruto BLIP captions images depict copyrighted characters
# (Masashi Kishimoto / Shueisha); the dataset card's license terms apply, and the fine-tuned model is for
# **non-commercial research only**.
#
# **NSFW filter:** the Stable Diffusion safety checker was kept enabled for every generation; flagged counts below.

# %%
gen_meta = pd.read_csv(GEN_DIR / "metadata.csv")
nsfw_tab = gen_meta.groupby(["model", "category"]).nsfw_flag.sum().unstack(fill_value=0)
nsfw_tab["total_flagged"] = nsfw_tab.sum(axis=1); nsfw_tab["images"] = gen_meta.groupby("model").size()
nsfw_tab

# %% [markdown]
# ### Memorization check
# For 10 in-domain images from the final run (seed 0), find the most similar **training** image by CLIP image-embedding
# cosine similarity, and show the 3 closest pairs.

# %%
if "clip_model" not in globals():   # after a restart
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

@torch.no_grad()
def clip_img_emb(paths, bs=32):
    embs = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        e = clip_model.get_image_features(**clip_proc(images=ims, return_tensors="pt").to(dev))
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(embs)

train_meta = [json.loads(l) for l in open(DATA_DIR / "train" / "metadata.jsonl")]
train_paths = [DATA_DIR / "train" / m["file_name"] for m in train_meta]
gen_paths = [GEN_DIR / FINAL_RUN / f"p{pi:02d}_s0.png" for pi in range(10)]   # first 10 prompts are in-domain
tr_e, ge_e = clip_img_emb(train_paths), clip_img_emb(gen_paths)
sims = ge_e @ tr_e.T
best_sim, best_idx = sims.max(dim=1)
mem_df = pd.DataFrame({"gen_image": [p.name for p in gen_paths], "nearest_train": [train_paths[i].name for i in best_idx],
                       "clip_cosine": best_sim.numpy().round(4)}).sort_values("clip_cosine", ascending=False)
print(mem_df.to_string(index=False))
top3 = mem_df.head(3)
fig, axes = plt.subplots(3, 2, figsize=(7, 10))
for r, row in enumerate(top3.itertuples()):
    axes[r, 0].imshow(Image.open(GEN_DIR / FINAL_RUN / row.gen_image)); axes[r, 0].set_title(f"generated {row.gen_image}", fontsize=9)
    axes[r, 1].imshow(Image.open(DATA_DIR / "train" / row.nearest_train)); axes[r, 1].set_title(f"train {row.nearest_train}  cos={row.clip_cosine:.3f}", fontsize=9)
    axes[r, 0].axis("off"); axes[r, 1].axis("off")
plt.tight_layout(); plt.savefig(GEN_DIR / "grids" / "memorization_top3.png", dpi=100); plt.show()
mem_df.to_csv(OUT / "memorization.csv", index=False)

# %% [markdown]
# **Memorization interpretation (your words):** TODO — e.g. are any of the top-3 pairs near-copies?

# %% [markdown]
# ## Push both LoRA adapters to Hugging Face (with a model card)
# Numbers are filled from the computed results; replace the TODO lines with your own words before pushing the final card.

# %%
from huggingface_hub import HfApi
api = HfApi()
train_df = pd.read_csv(OUT / "train_runs.csv").set_index("run")
res_df = pd.read_csv(OUT / "results.csv").set_index("model")
ds_sum = json.load(open(OUT / "data_summary.json"))

def model_card(name):
    t, r = train_df.loc[name], res_df.loc[name]
    return f"""---
license: other
base_model: {BASE_MODEL}
datasets: [{DATASET_ID}]
tags: [stable-diffusion, lora, text-to-image, diffusers]
---
# {HF_LORA_REPOS[name].split('/')[-1]} — DATA 266 Lab 2 Part 2 (team {TEAM})

**Base model:** `{BASE_MODEL}` · **Method:** LoRA rank {RUNS[name]} on the UNet attention layers (diffusers `train_text_to_image_lora.py` v0.31.0)
{"**Selected as the final run.**" if name == FINAL_RUN else ""}

## Data
`{DATASET_ID}`: {ds_sum['clean']} clean images after removing {ds_sum['corrupted_removed']} corrupted and
{ds_sum['exact_dup_removed'] + ds_sum['near_dup_removed']} duplicate images; 90/10 split (seed 266) → {ds_sum['train']} train / {ds_sum['test']} test; {RES}×{RES}.

## Training settings
| trainable params | lr | steps | batch (effective) | precision | peak GPU mem | time | GPU |
|---|---|---|---|---|---|---|---|
| {int(t.trainable_params):,} | {t.learning_rate} | {int(t.steps)} | {int(t.effective_batch)} | fp16 | {t.peak_gpu_mem_GiB} GiB | {t.train_time_min} min ({t.gpu_hours} GPU-h) | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'T4'} |

## Results (20 fixed prompts × 2 seeds; DPM-Solver++ 30 steps, CFG 7.5)
| model | KID (in-domain) ↓ | CLIP score ↑ | LPIPS diversity |
|---|---|---|---|
| base | {res_df.loc['base','KID_mean(in-domain)']:.4f} | {res_df.loc['base','CLIP_score(all 40)']:.2f} | {res_df.loc['base','LPIPS_diversity']:.3f} |
| this model | {r['KID_mean(in-domain)']:.4f} | {r['CLIP_score(all 40)']:.2f} | {r['LPIPS_diversity']:.3f} |

## Usage
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("{BASE_MODEL}").to("cuda")
pipe.load_lora_weights("{HF_LORA_REPOS[name]}")
```

## Limitations
TODO (your own words)

## License
Images in the training set depict copyrighted Naruto characters. This adapter is released for **non-commercial research and
education only**; the base model is under the CreativeML OpenRAIL-M license.
"""

for name in RUNS:
    d = RUNS_DIR / name
    (d / "README.md").write_text(model_card(name))
    api.create_repo(HF_LORA_REPOS[name], repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(d), repo_id=HF_LORA_REPOS[name], repo_type="model",
                      allow_patterns=["pytorch_lora_weights.safetensors", "README.md", "timing.json"])
    print("pushed:", f"https://huggingface.co/{HF_LORA_REPOS[name]}")

# %% [markdown]
# ## Save all numbers (single source for the report)

# %%
gen_seconds = json.load(open(OUT / "gen_seconds_per_image.json")) if (OUT / "gen_seconds_per_image.json").exists() else {}
metrics = {"commit": commit, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED,
           "data": json.load(open(OUT / "data_summary.json")),
           "training": pd.read_csv(OUT / "train_runs.csv").to_dict(orient="records"),
           "results": pd.read_csv(OUT / "results.csv").to_dict(orient="records"),
           "gen_seconds_per_image": gen_seconds,
           "human_eval": human if "human" in globals() else None,
           "memorization_top3": pd.read_csv(OUT / "memorization.csv").head(3).to_dict(orient="records"),
           "final_run": FINAL_RUN, "hf_models": HF_LORA_REPOS, "hf_dataset": HF_GEN_DATASET}
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2, default=str)
print(json.dumps(metrics, indent=2, default=str)[:3000])
