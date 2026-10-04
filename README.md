# ZETA-IoT — Evaluation Artifacts (anonymized)

Artifacts for "ZETA-IoT: A Zero-Trust Ensemble Architecture with Tiered AI
Governance for Resource-Constrained IoT Edge Security" (IoT 2026 submission).

## Contents
- `main_revised.tex`, `refs.bib` — revised manuscript source (ACM sigconf, double-blind)
- `figures/` — figures used in the paper
- `experiment_artifacts/` — all evaluation code and results:
- `eval5.py` — main evaluation (produces FINAL_RESULTS.json; every reported number traces to it)
- `prep_npz.py`, `extract_rars.py` — data preparation from the public N-BaIoT
release (UCI ML Repository, dataset 442, DOI 10.24432/C5RC8J)
- `abl.py` — component/ensemble ablation
- `matched_fpr.py` — matched-FPR comparison of components and fusion rules
- `multidevice_batch.py` — multi-device batching benchmark
- `FINAL_RESULTS.json` — per-device and aggregate results

## Requirements
python3, numpy, pandas, scikit-learn (psutil optional, for memory reporting)

## Run order
1. Download the N-BaIoT dataset (UCI dataset 442) and extract the zip.
2. `python extract_rars.py all`
3. `python prep_npz.py`
4. `python eval5.py all` then `python eval5.py finish`
5. `python matched_fpr.py all`
6. `python multidevice_batch.py all 4096 20000`
