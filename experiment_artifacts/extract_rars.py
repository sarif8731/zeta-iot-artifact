#!/usr/bin/env python3
import glob, io, os, sys, zipfile
import pandas as pd
from unrar.cffi import rarfile

ZIP = "/sessions/dazzling-lucid-cori/mnt/IOT paper/detection+of+iot+botnet+attacks+n+baiot.zip"
OUT = os.path.expanduser("~/nbaiot")
CAP = 15000
devices = sys.argv[1:] if len(sys.argv) > 1 else None

zf = zipfile.ZipFile(ZIP)
rars = [n for n in zf.namelist() if n.endswith(".rar")]
for name in rars:
    dev = name.split("/")[0]
    if devices and dev not in devices:
        continue
    fam = "mirai" if "mirai" in name else "gafgyt"
    local_rar = os.path.join(OUT, dev, os.path.basename(name))
    os.makedirs(os.path.join(OUT, dev), exist_ok=True)
    if not os.path.exists(local_rar):
        with zf.open(name) as src, open(local_rar, "wb") as dst:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                dst.write(b)
    rf = rarfile.RarFile(local_rar)
    members = [m for m in rf.namelist() if m.endswith(".csv")]
    for m in members:
        atk = os.path.basename(m).replace(".csv", "")
        out_csv = os.path.join(OUT, dev, fam + "_" + atk + ".csv")
        if os.path.exists(out_csv):
            continue
        with rf.open(m) as f:
            raw = f.read()
        lines = raw.split(b"\n")
        header, body = lines[0], [l for l in lines[1:] if l]
        if len(body) > CAP:
            stride = len(body) / CAP
            body = [body[int(i * stride)] for i in range(CAP)]
        df = pd.read_csv(io.BytesIO(b"\n".join([header] + body)))
        df.to_csv(out_csv, index=False)
        print(dev, fam, atk, len(df), flush=True)
    done = all(os.path.exists(os.path.join(OUT, dev, fam + "_" + os.path.basename(m).replace(".csv", "") + ".csv")) for m in members)
    if done:
        os.remove(local_rar)
print("DONE")
