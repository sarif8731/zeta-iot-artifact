#!/usr/bin/env python3
"""Multi-device batching benchmark (Reviewer 1, Major 4).

Measures whether the single-model batch throughput survives when each device needs a
DIFFERENT model: trains all per-device ensembles (eval5.py recipe), then replays an
interleaved multi-device arrival stream, dispatching per-device batches, and reports
sustained throughput, batch waiting latency, and peak resident memory.

Usage:  python multidevice_batch.py all [BATCH_SIZE] [N_SNAPSHOTS_PER_DEVICE]
Output: ~/state5/multidevice_batch.json
Requires: psutil for RSS on Linux (falls back to tracemalloc if unavailable).
"""
import glob, json, os, sys, time, tracemalloc
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HOME = os.path.expanduser("~"); DATA = os.path.join(HOME, "nbaiot"); ST = os.path.join(HOME, "state5")
os.makedirs(ST, exist_ok=True)
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N_PER_DEV = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
try:
    import psutil
    RSS = True
except ImportError:
    RSS = False

devs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.npz"))) \
    if (len(sys.argv) > 1 and sys.argv[1] == "all") else sys.argv[1:2] or None
if not devs:
    sys.exit("no devices")

def scores_of(m, X):
    return np.column_stack([-m["IF"].decision_function(X), -m["OCSVM"].decision_function(X),
                            np.mean((X - m["AE"].predict(X)) ** 2, axis=1)])

models, held = {}, {}
t0 = time.time()
for dev in devs:
    z = np.load(os.path.join(DATA, dev + ".npz")); d = {k: z[k] for k in z.files}
    rng = np.random.RandomState(42); ben = d["benign"].copy(); rng.shuffle(ben)
    Xtr = StandardScaler().fit(ben[:int(.6 * len(ben))]).transform(ben[:int(.6 * len(ben))])
    models[dev] = {
        "IF": IsolationForest(n_estimators=200, max_samples=1024, random_state=42, n_jobs=-1).fit(Xtr),
        "OCSVM": make_pipeline(Nystroem(gamma=0.1, n_components=100, random_state=42),
                               SGDOneClassSVM(nu=0.05, random_state=42)).fit(Xtr),
        "AE": MLPRegressor(hidden_layer_sizes=(48, 8, 48), max_iter=120, random_state=42).fit(Xtr, Xtr)}
    held[dev] = StandardScaler().fit(ben[:int(.6 * len(ben))]).transform(
        ben[int(.8 * len(ben)):int(.8 * len(ben)) + N_PER_DEV])
train_s = round(time.time() - t0, 1)

process = psutil.Process() if RSS else None
tracemalloc.start()
rss0 = process.memory_info().rss / 1e6 if RSS else 0

# --- single-model reference (device 0 only), batch B ---
d0 = devs[0]
Xb = held[d0][:B]
t0 = time.time()
for _ in range(20):
    scores_of(models[d0], Xb)
single_ms = (time.time() - t0) / 20 * 1000

# --- multi-device interleaved arrival ---
# round-robin arrival: 1 snapshot per device in turn => per-device queues fill uniformly
queues = {dev: [] for dev in devs}
n_batches = N_PER_DEV // B
t0 = time.time()
wait_samples = 0  # snapshots that waited in queue before their batch ran
for b in range(n_batches):
    for dev in devs:
        q = queues[dev]
        q.extend(held[dev][b * B:(b + 1) * B])
        if len(q) >= B:
            batch = np.asarray(q[:B]); del q[:B]
            wait_samples += len(q)  # remaining queued snapshots waited one round
            scores_of(models[dev], batch)
wall = time.time() - t0
total = n_batches * B * len(devs)
peak_py, _ = tracemalloc.get_traced_memory()
rss1 = process.memory_info().rss / 1e6 if RSS else 0

out = {"batch_size": B, "snapshots_per_device": N_PER_DEV, "n_devices": len(devs),
       "devices": devs, "train_time_all_s": train_s,
       "single_model": {"batch_ms_mean": round(single_ms, 3),
                        "ms_per_snapshot": round(single_ms / B, 5),
                        "snapshots_per_s": int(B / (single_ms / 1000))},
       "multidevice": {"wall_s": round(wall, 2), "total_snapshots": total,
                       "ms_per_snapshot": round(wall / total * 1000, 5),
                       "snapshots_per_s": int(total / wall),
                       "queued_snapshots_total": wait_samples,
                       "peak_python_alloc_MB": round(peak_py / 1e6, 1),
                       "rss_delta_MB": round(rss1 - rss0, 1) if RSS else None},
       "note": "round-robin interleave; queues drain when a full batch accumulates. "
               "Compare snapshots_per_s: single-model vs multi-device dispatch."}
json.dump(out, open(os.path.join(ST, "multidevice_batch.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
