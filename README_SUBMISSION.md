# IoT 2026 Submission Package — ZETA-IoT

Deadline: **30 July 2026 (AoE)** · Submit via EasyChair: https://easychair.org/conferences/?conf=iot2026

## What's in this folder

- `main.tex` — the full anonymized manuscript (ACM sigconf format, double-blind ready)
- `refs.bib` — bibliography (all 20 references verified real)
- `figures/` — fig_architecture.pdf, fig_per_attack.pdf, fig_per_device.pdf (generated from real results)
- `preview_layout_check.pdf` — approximate layout preview (NOT for submission; ~5-6 pages, well under the 8-page limit)
- `../experiment/` — reproducibility package: evaluation code + FINAL_RESULTS.json (every number in the paper traces to this file)

## How to produce the submission PDF (≈5 minutes)

1. Go to https://www.overleaf.com → New Project → Templates → search "ACM Conference Proceedings Primary Article Template" (acmart).
2. In the template, delete the sample content of `sample-sigconf.tex` and paste in the contents of `main.tex` (or upload `main.tex` and set it as the main document).
3. Upload `refs.bib` and the `figures/` folder (keep the folder name `figures`).
4. Compile → download PDF.

The document class line is already set for double-blind review:
`\documentclass[sigconf,review,anonymous]{acmart}`

## Before you click submit — checklist

- [ ] PDF is anonymous: no name, affiliation, email, ORCID anywhere (already handled in main.tex — do not add them back)
- [ ] ≤ 8 pages excluding references
- [ ] ACM authorship / generative-AI policy: disclose AI assistance in drafting per https://www.acm.org/publications/policies/new-acm-policy-on-authorship (EasyChair may have a field for this; if not, follow the policy's disclosure instructions)
- [ ] Do NOT upload the old cover letter (it identifies you and isn't required)
- [ ] Camera-ready (after acceptance, due 2 Oct 2026): remove `review,anonymous` options, restore author block, add funding/acks

## Honest-scope notes (important if reviewers ask)

- All quantitative claims cover the Layer-3 detection/routing evaluation on N-BaIoT. Layers 1/2/4 are presented as design only.
- Latency/memory were measured on x86 CPU (stated in the paper); no Raspberry Pi/FortiGate/TPM measurements are claimed.
- The full per-device results are in `../experiment/FINAL_RESULTS.json`; the code that produced them is `../experiment/eval5.py` (plus `extract_rars.py`, `prep_npz.py` for data prep and `abl.py` for the ablation). Releasing these as open source on acceptance backs the paper's reproducibility commitment.
