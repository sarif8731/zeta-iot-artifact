import glob, os, sys
import numpy as np, pandas as pd
OUT = os.path.expanduser("~/nbaiot")
CAP = 15000
rng = np.random.RandomState(42)
for dev_dir in sorted(glob.glob(os.path.join(OUT, "*/"))):
    dev = os.path.basename(dev_dir.rstrip("/"))
    npz = os.path.join(OUT, dev + ".npz")
    if os.path.exists(npz):
        continue
    arrs = {}
    for csv in sorted(glob.glob(os.path.join(dev_dir, "*.csv"))):
        label = os.path.basename(csv).replace(".csv", "").replace("_traffic", "")
        df = pd.read_csv(csv)
        X = df.values.astype(np.float32)
        if len(X) > CAP:
            X = X[rng.choice(len(X), CAP, replace=False)]
        arrs[label] = X
    np.savez_compressed(npz, **arrs)
    print(dev, {k: v.shape for k, v in arrs.items()}, flush=True)
print("PREP DONE")
