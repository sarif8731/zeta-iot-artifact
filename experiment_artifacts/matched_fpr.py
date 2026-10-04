#!/usr/bin/env python3
"""Matched-FPR comparison of ensemble components and fusion rules (Reviewer 1, Major 2).

Trains the exact ZETA-IoT per-device pipeline (same seeds/hyperparameters as eval5.py),
then, at MATCHED window-level false-positive budgets on benign test data, reports:
  - TDR of each component (IF, OCSVM, AE), max fusion, and mean fusion
  - missed-attack overlap between fusion rules (windows caught by one rule, missed by other)
  - scoring cost per rule (relative to single-component scoring)

Usage:  python matched_fpr.py all            # all devices found in ~/nbaiot
        python matched_fpr.py Danmini_Doorbell Ecobee_Thermostat ...
Output: ~/state5/matched_fpr.json
"""
import glob, json, os, time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HOME = os.path.expanduser("~"); DATA = os.path.join(HOME, "nbaiot"); ST = os.path.join(HOME, "state5")
os.makedirs(ST, exist_ok=True)
W = 11
BUDGETS = [0.01, 0.0145]  # nominal 1%, and the paper's measured ensemble FPR for reference

def win_med(z, w=W):
    k = len(z) // w
    return np.median(z[:k * w].reshape(k, w), axis=1)

devs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.npz"))) \
    if (len(os.sys.argv) > 1 and os.sys.argv[1] == "all") else os.sys.argv[1:]

per_dev = {}
for dev in devs:
    z = np.load(os.path.join(DATA, dev + ".npz")); d = {k: z[k] for k in z.files}
    rng = np.random.RandomState(42); ben = d["benign"].copy(); rng.shuffle(ben)
    n = len(ben); ntr, nva = int(.6 * n), int(.2 * n)
    scaler = StandardScaler().fit(ben[:ntr])
    Xtr, Xva, Xte = (scaler.transform(a) for a in (ben[:ntr], ben[ntr:ntr + nva], ben[ntr + nva:]))
    m = {"IF": IsolationForest(n_estimators=200, max_samples=1024, random_state=42, n_jobs=-1).fit(Xtr),
         "OCSVM": make_pipeline(Nystroem(gamma=0.1, n_components=100, random_state=42),
                                SGDOneClassSVM(nu=0.05, random_state=42)).fit(Xtr),
         "AE": MLPRegressor(hidden_layer_sizes=(48, 8, 48), max_iter=120, random_state=42).fit(Xtr, Xtr)}
    def scores(X):
        return np.column_stack([-m["IF"].decision_function(X), -m["OCSVM"].decision_function(X),
                                np.mean((X - m["AE"].predict(X)) ** 2, axis=1)])
    Sva = scores(Xva); mu, sd = Sva.mean(0), Sva.std(0) + 1e-9
    zstd = lambda X: (scores(X) - mu) / sd
    # window-level score streams per rule
    rules = ["IF", "OCSVM", "AE", "max", "mean"]
    wva = {r: win_med(zstd(Xva)[:, 0 if r == "IF" else 1 if r == "OCSVM" else 2].ravel()
                      if r in ("IF", "OCSVM", "AE") else
                      (zstd(Xva).max(1) if r == "max" else zstd(Xva).mean(1))) for r in rules}
    wte = {r: win_med(zstd(Xte)[:, 0 if r == "IF" else 1 if r == "OCSVM" else 2].ravel()
                      if r in ("IF", "OCSVM", "AE") else
                      (zstd(Xte).max(1) if r == "max" else zstd(Xte).mean(1))) for r in rules}
    atk = {r: [] for r in rules}
    for lbl, X in sorted(d.items()):
        if lbl == "benign": continue
        Z = zstd(X)
        for r in rules:
            s = Z[:, 0 if r == "IF" else 1 if r == "OCSVM" else 2] if r in ("IF", "OCSVM", "AE") \
                else (Z.max(1) if r == "max" else Z.mean(1))
            atk[r].append(win_med(s))
    atk = {r: np.concatenate(v) for r, v in atk.items()}
    row = {"budgets": {}}
    for b in BUDGETS:
        entry = {}
        for r in rules:
            thr = float(np.quantile(wva[r], 1 - b))
            fpr = float((wte[r] > thr).mean())
            tdr = float((atk[r] > thr).mean())
            entry[r] = {"thr": round(thr, 4), "fpr_achieved": round(fpr, 4), "tdr": round(tdr, 4)}
        # missed-attack overlap between fusion rules at this budget
        thr_max = np.quantile(wva["max"], 1 - b); thr_mean = np.quantile(wva["mean"], 1 - b)
        c_max, c_mean = atk["max"] > thr_max, atk["mean"] > thr_mean
        n_win = len(c_max)
        entry["overlap"] = {
            "n_attack_windows": int(n_win),
            "caught_max_missed_mean": int((c_max & ~c_mean).sum()),
            "caught_mean_missed_max": int((c_mean & ~c_max).sum()),
            "caught_both": int((c_max & c_mean).sum()),
            "missed_both": int((~c_max & ~c_mean).sum()),
        }
        row["budgets"][str(b)] = entry
    per_dev[dev] = row
    print(dev, json.dumps({b: {r: v["tdr"] for r, v in e.items() if r != "overlap"}
                           for b, e in row["budgets"].items()}), flush=True)

# aggregate (micro over devices, weighting by attack-window counts)
agg = {}
for b in BUDGETS:
    bs = str(b); tot = {"caught_max_missed_mean": 0, "caught_mean_missed_max": 0, "n": 0}
    for dev, row in per_dev.items():
        o = row["budgets"][bs]["overlap"]
        for k in ("caught_max_missed_mean", "caught_mean_missed_max"):
            tot[k] += o[k]
        tot["n"] += o["n_attack_windows"]
    agg[bs] = {**{k: tot[k] for k in ("caught_max_missed_mean", "caught_mean_missed_max")},
               "n_attack_windows": tot["n"],
               "pct_caught_max_missed_mean": round(tot["caught_max_missed_mean"] / max(tot["n"], 1), 4),
               "pct_caught_mean_missed_max": round(tot["caught_mean_missed_max"] / max(tot["n"], 1), 4)}
out = {"protocol": "identical to eval5.py; thresholds set on benign validation per rule at matched window-FPR budget; TDR on all attack windows",
       "W": W, "budgets": BUDGETS, "cost_note": "max/mean fusion cost = sum of 3 component scorings + argmax/mean; components cost 1/3 of fusion",
       "per_device": per_dev, "aggregate_overlap": agg}
json.dump(out, open(os.path.join(ST, "matched_fpr.json"), "w"), indent=1)
print(json.dumps(agg, indent=1))
