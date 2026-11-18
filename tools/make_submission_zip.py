"""Build Pair_Programming_Team_06.zip for Canvas (lab section S.1).

Usage (from the repo root):
    python tools/make_submission_zip.py [--out ~/Downloads/Pair_Programming_Team_06.zip] --weights <dir> [--allow-missing-report]

Contents: the report PDF, CONTRIBUTIONS.md, AI_USAGE.md, links.txt, and Part1/Part2/Part3 folders with each part's
executed notebooks and output files. Only files tracked by git are packed (so secrets, weights and the team member
photo can never end up in the zip); Part 2 LoRA weights are added from --weights (not from git).
"""
import argparse, os, subprocess, sys, zipfile

REPORT = "report/Pair_Programming_Team_06_Report.pdf"
PARTS = {
    "Part1": ["notebooks/p1_rag.ipynb", "notebooks/p1_human_eval.ipynb", "part1/"],
    "Part2": ["notebooks/p2_finetune.ipynb", "part2/"],
    "Part3": ["notebooks/p3_agents.ipynb", "notebooks/p3_human_eval.ipynb", "part3/"],
}
WEIGHTS = {"Part2": "Model weights: Part2/weights/lora_r4 (run 1) and Part2/weights/lora_r32 (run 2, final) hold the LoRA adapters\n"
                    "(pytorch_lora_weights.safetensors, load with pipe.load_lora_weights on stable-diffusion-v1-5).\n"
                    "Same files on Hugging Face - see links.txt.\n",
           "Part3": "Part 3 loads the Part 2 r=32 LoRA from Hugging Face (links.txt); no other trained weights.\n",
           "Part1": "Part 1 uses only pretrained open models (no trained weights). Kaggle notebook: see links.txt.\n"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/Downloads/Pair_Programming_Team_06.zip"))
ap.add_argument("--allow-missing-report", action="store_true")
ap.add_argument("--weights", help="folder holding lora_r4/ and lora_r32/ (downloaded from Hugging Face; weights are never committed)")
a = ap.parse_args()

tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n"))
if not os.path.exists(REPORT) and not a.allow_missing_report:
    sys.exit(f"Missing {REPORT}. Export the report PDF there first (or pass --allow-missing-report for a dry run).")

n = 0
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    if os.path.exists(REPORT):
        z.write(REPORT, "Pair_Programming_Team_06_Report.pdf"); n += 1
    for f in ("CONTRIBUTIONS.md", "AI_USAGE.md", "links.txt"):
        z.write(f, f); n += 1
    for part, items in PARTS.items():
        for it in items:
            files = sorted(t for t in tracked if t == it or (it.endswith("/") and t.startswith(it)))
            for f in files:
                z.write(f, f"{part}/{f.split('/', 1)[1] if f.startswith(('notebooks/', 'part')) else f}"); n += 1
        if part == "Part2":
            if not a.weights:
                sys.exit("Part 2 weights are required in the zip: pass --weights <folder with lora_r4/ and lora_r32/>.")
            for run in ("lora_r4", "lora_r32"):
                for fn in ("pytorch_lora_weights.safetensors", "README.md", "timing.json"):
                    z.write(os.path.join(a.weights, run, fn), f"Part2/weights/{run}/{fn}"); n += 1
        z.writestr(f"{part}/WEIGHTS_AND_LINKS.txt", WEIGHTS[part]); n += 1
print(f"wrote {a.out}: {n} files, {os.path.getsize(a.out) / 1e6:.1f} MB"
      + ("" if os.path.exists(REPORT) else "  ** WITHOUT the report PDF - not ready to submit **"))
