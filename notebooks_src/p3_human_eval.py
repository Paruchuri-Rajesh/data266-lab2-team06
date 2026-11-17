# %% [markdown]
# # Part 3 - human judgements and final metrics
# Both team members judged every scenario output (15 scenarios x with/without critique) **independently** against its
# rubric (`pass` 0/1, plus agreement with the critic's decision where one was made). Disagreements were resolved by the
# team into `final_pass`. This notebook only aggregates the committed CSVs from the T4 evaluation run
# (`part3/results_raw.csv`, `part3/redteam_results.csv`) and `part3/human_judgements.csv`; it uses the same formulas as
# Section 9 of `p3_agents.ipynb`. Runs on CPU in seconds; no model is loaded.

# %%
# Run information (required)
import subprocess, sys, random, json, os, numpy as np, pandas as pd
SEED = 266; random.seed(SEED); np.random.seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print({"commit": commit or "no git checkout", "python": sys.version.split()[0], "gpu": "none (CPU only)", "seed": SEED})
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

# %%
res = pd.read_csv("part3/results_raw.csv")
rt = pd.read_csv("part3/redteam_results.csv")
hj = pd.read_csv("part3/human_judgements.csv")
scen = {s["id"]: s for s in json.load(open("part3/scenarios.json"))["scenarios"]}
assert len(res) == 30 and len(rt) == 10 and len(hj) == 60 and hj["pass"].notna().all() and hj["final_pass"].notna().all()
print("rows:", {"scenario runs": len(res), "red-team cases": len(rt), "judgements": len(hj)})

# %%
# Metrics (same definitions as p3_agents.ipynb Section 9)
metrics = {"n_scenarios": len(scen), "redteam_attack_success_rate": f"{int(rt.attack_succeeded_auto.sum())}/10"}
for uc in (False, True):
    key = "with_critique" if uc else "without_critique"
    sub = res[res.use_critique == uc]
    h = hj[hj.use_critique == uc]
    piv = h.pivot_table(index="scenario_id", columns="rater", values="pass")
    final = h.groupby("scenario_id")["final_pass"].first()
    m = {"median_seconds_per_request": float(sub.seconds.median()),
         "auto_route_accuracy": f"{int(sub.route_ok.sum())}/15",
         "inter_rater_agreement": round(float((piv.nunique(axis=1) == 1).mean()), 3),
         "task_success_rate": f"{int(final.sum())}/15"}
    if uc and h["critic_decision_agree"].notna().any():
        m["critic_agreement"] = round(float(h["critic_decision_agree"].mean()), 3)
    metrics[key] = m
json.dump(metrics, open("part3/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))

# %%
# Results table for the report (baseline = without critique, final = with critique)
def tbl(m):
    return {"Task success (/15)": m["task_success_rate"], "Critic agreement": m.get("critic_agreement", "n/a"),
            "Red-team attack success (/10)": metrics["redteam_attack_success_rate"],
            "Human inter-rater agreement": m["inter_rater_agreement"],
            "Median s / request": round(m["median_seconds_per_request"], 2)}
pd.DataFrame({"Baseline (no critique)": tbl(metrics["without_critique"]),
              "Final (with critique)": tbl(metrics["with_critique"])})

# %%
# Task success by category (final_pass)
f = hj.drop_duplicates(["scenario_id", "use_critique"]).copy()
f["category"] = f.scenario_id.map(lambda s: scen[s]["category"])
f.pivot_table(index="category", columns="use_critique", values="final_pass", aggfunc="sum").rename(
    columns={False: "without critique (/3)", True: "with critique (/3)"})

# %%
# Where the two raters disagreed (resolved into final_pass)
piv = hj.pivot_table(index=["scenario_id", "use_critique"], columns="rater", values="pass").astype(int)
fin = hj.drop_duplicates(["scenario_id", "use_critique"]).set_index(["scenario_id", "use_critique"])["final_pass"]
dis = piv[piv.member_A != piv.member_B].join(fin)
print(f"{len(dis)} of {len(piv)} judgements differ between raters")
dis

# %%
# Critic: per-rater agreement with the critic's accept/revise decision, and the critic's decisions vs final_pass
crit = hj[hj.critic_decision_agree.notna()]
print(crit.groupby("rater").critic_decision_agree.mean().round(3).to_dict())
cr = res[res.use_critique & res.critic_decision.isin(["accept", "revise"])][["scenario_id", "critic_decision", "clip_score", "revisions"]]
cr = cr.merge(fin.xs(True, level="use_critique").rename("final_pass"), left_on="scenario_id", right_index=True)
cr

# %%
# Red-team cases (automatic check: attack succeeds if the system did not behave as expected)
rt[["case_id", "attack_type", "expected", "route", "attack_succeeded_auto"]]
