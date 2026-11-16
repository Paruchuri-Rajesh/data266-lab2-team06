# %% [markdown]
# # Part 1 - human groundedness evaluation
# Both team members scored every final-system (hybrid) answer on the golden set **independently**
# (1 = not supported by the cited evidence, 5 = fully supported). This notebook only aggregates `part1/human_eval.csv`.
# Runs on CPU in seconds; no model is loaded.

# %%
# Run information (required)
import subprocess, sys, random, numpy as np, pandas as pd, os
SEED = 266; random.seed(SEED); np.random.seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print({"commit": commit or "no git checkout", "python": sys.version.split()[0], "gpu": "none (CPU only)", "seed": SEED})
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

# %%
he = pd.read_csv("part1/human_eval.csv")
gold = pd.read_csv("part1/golden_set.csv", dtype=str)[["gid", "category", "gold_status"]]
res = pd.read_csv("part1/golden_results_hybrid.csv", dtype=str)[["gid", "correct"]]
he = he.merge(gold, on="gid").merge(res, on="gid")
a, b = he["rater_A"].astype(float), he["rater_B"].astype(float)
he["mean"] = (a + b) / 2
print(f"n = {len(he)} answers")
print(f"Groundedness (1-5): rater A {a.mean():.2f}, rater B {b.mean():.2f}, overall {he['mean'].mean():.2f}")

# %%
# Inter-rater agreement
def quadratic_kappa(x, y, k=5):
    x, y = np.asarray(x, int) - 1, np.asarray(y, int) - 1
    O = np.zeros((k, k))
    for i, j in zip(x, y):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()

print(f"Exact agreement: {(a == b).mean():.2f}")
print(f"Within 1 point: {((a - b).abs() <= 1).mean():.2f}")
print(f"Pearson r: {np.corrcoef(a, b)[0, 1]:.2f}")
print(f"Quadratic-weighted Cohen's kappa: {quadratic_kappa(a, b):.2f}")

# %%
# Groundedness by question category and by automatic correctness
print(he.groupby("category")[["rater_A", "rater_B", "mean"]].mean().round(2))
print()
print(he.groupby("correct")[["rater_A", "rater_B", "mean"]].agg(["mean", "count"]).round(2))

# %%
# Largest disagreements (useful to discuss in the report)
he["diff"] = (a - b).abs()
print(he.sort_values("diff", ascending=False)[["gid", "category", "pred_answer", "rater_A", "rater_B"]].head(6).to_string(index=False))
