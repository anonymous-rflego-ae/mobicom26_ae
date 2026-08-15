# RF-LEGO — Artifact Evaluation (AE)

This directory contains a self-contained, deterministic, one-click artifact-evaluation workflow for **RF-LEGO**, a modular signal processing-deep learning co-design framework for RF sensing via deep unrolling. It demonstrates four things:

1. **Artifact functionality**: installation, model initialization, a forward/backward pass, and shipped-weight inference for each of the three RF-LEGO modules.
2. **Result evaluation**: side-by-side, quantitative module-level results comparing RF-LEGO with the corresponding classical signal processing baselines.
3. **AE benchmark**: modality-specific held-out `.npz` files under `ae_data/`, using model-ready real-world inputs and ground truth for each RF-LEGO module.
4. **Reduced cascadability check** (related to paper Sec. 5.2.2): comparing classical front ends→CFAR with RF-LEGO range/Doppler FT→Detector and angle Beamformer→Detector pipelines.

The three module-level result groups:

| Module | Baseline | Metric |
|------|---|---|
| Frequency Transform | Bluestein FFT | PSLR / PAPR improvement (dB) |
| Beamformer | LASSO Beamformer | angle-MAE reduction (%) |
| Detector | CA-CFAR (nominal `P_FA` of `10^-3`) | exact-bin DR at method-native fixed thresholds |

Current prepared AE files:

| Modality | Module/task | File |
|---|---|---|
| mmWave | Beamformer | `ae_data/mmwave_beamformer.npz` |
| mmWave | Doppler FT | `ae_data/mmwave_dopplerft.npz` |
| mmWave | Range FT | `ae_data/mmwave_rangeft.npz` |
| UWB | Detector | `ae_data/uwb_detector.npz` |
| UWB | Doppler FT | `ae_data/uwb_dopplerft.npz` |
| WiFi | Doppler FT | `ae_data/wifi_dopplerft.npz` |

The mmWave Beamformer file contains 50 prepared frames from the processed real
measurement collection, spanning angles from -60° to +60°.

---

## One-click Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anonymous-rflego-ae/mobicom26_ae/blob/main/ae/RF_LEGO_AE_Colab.ipynb)

Open `ae/RF_LEGO_AE_Colab.ipynb` and run all cells. The notebook uses the
prepared modality `.npz` files and shipped pretrained weights, then evaluates
and plots the results (roughly **3 minutes** on Colab CPU).

---

## Local workflow (Optional)

From the repo root:

```bash
# 1. install the package + AE dependencies
pip install -e .
pip install -r ae/requirements.txt

# 2. evaluate all modality/task files -> ae/results/metrics/*.json
python ae/scripts/evaluate.py --module all

# 3. render one fixed single-panel visualization per dataset -> ae/results/figures/*.png
python ae/scripts/plot_reproduction.py

# 4. cascadability: RF-LEGO vs classical front ends -> ae/results/metrics/cascade.json
python ae/scripts/cascade.py
```

Reviewers do not need to train anything: every step above runs on the shipped
pretrained weights under `ae/results/weights/`.

The Detector corrects the inherited scalar-expand → pre-LayerNorm bug. Its first
state-space layer now receives the raw broadcast profile, so A/B have finite,
nonzero sample-conditioned gradients on the first backward pass. Later layers
may use pre-normalization only after the hidden features have become
heterogeneous, while the outer residual preserves the direct scalar path. The
input-path, direct-path, and temporal-memory behaviors are regression tested.

## Expected runtime & requirements

- **Hardware:** CPU is sufficient; shipped-weight evaluation runs on CPU for portability.
- **Software:** Python 3.10-3.12. `ae/requirements.txt` pins the AE runtime dependencies, and `uv.lock` records the package-resolution lockfile used for this artifact.
- **Runtime:** shipped-weight evaluation end-to-end ≈ 2-3 min.
- **Strict determinism:** set `RFLEGO_STRICT_DETERMINISM=1` to request PyTorch deterministic algorithms. The default keeps seeded CPU evaluation fast across newer PyTorch builds.

---

### Output files

- `ae/results/metrics/<dataset>.json`: baseline vs. RF-LEGO metrics and improvement.
- `ae/results/metrics/summary.json`: modality/module/result summary rows.
- `ae/results/figures/result_<dataset>.png`: one fixed single-panel visualization per dataset.
- `ae/results/cache/<dataset>_plotcache.npz`: cached arrays the plots are drawn from.
- `ae/results/metrics/cascade.json`: cascadability results (RF-LEGO vs. classical front ends).
- `ae/results/env/environment.json`: runtime/environment snapshot.

## Reported Results

| Modality | Module/task | Typical measured |
|---|---|---|
| mmWave | Beamformer | angle-MAE reduction +70.0 % |
| mmWave | Doppler FT | PSLR improvement +21.6 dB, PAPR improvement +4.6 dB |
| mmWave | Range FT | PSLR improvement +25.6 dB, PAPR improvement +3.3 dB |
| UWB | Detector | RF default DR 1.000; CA-CFAR DR 1.000 |
| UWB | Doppler FT | PSLR improvement +21.8 dB, PAPR improvement +4.6 dB |
| WiFi | Doppler FT | PSLR improvement +1.2 dB, PAPR improvement +5.9 dB |

## Paper-to-artifact result map

The table below identifies the exact command, output file, and JSON key used to
check each shipped quantitative result. Values are for the pretrained weights in
`ae/results/weights/`.

| Result | Command | Output JSON key | Expected value |
|---|---|---|---|
| mmWave Beamformer angle-MAE reduction | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/mmwave_beamformer.json: improvement.mae_reduction_percent` | `+70.0000` |
| mmWave Doppler FT PSLR / PAPR gain | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/mmwave_dopplerft.json: improvement.pslr_db`, `improvement.papr_db` | `+21.6262 dB`, `+4.6224 dB` |
| mmWave Range FT PSLR / PAPR gain | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/mmwave_rangeft.json: improvement.pslr_db`, `improvement.papr_db` | `+25.6448 dB`, `+3.2862 dB` |
| UWB Detector method-native operating points | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/uwb_detector.json: baseline.dr`, `rflego.dr` | `1.0000`, `1.0000` |
| UWB Doppler FT PSLR / PAPR gain | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/uwb_dopplerft.json: improvement.pslr_db`, `improvement.papr_db` | `+21.8045 dB`, `+4.6431 dB` |
| WiFi Doppler FT PSLR / PAPR gain | `python ae/scripts/evaluate.py --module all` | `ae/results/metrics/wifi_dopplerft.json: improvement.pslr_db`, `improvement.papr_db` | `+1.2028 dB`, `+5.8527 dB` |
| Cascadability range / Doppler / angle DR | `python ae/scripts/cascade.py` | `ae/results/metrics/cascade.json: rows[*].dr_classical`, `dr_rflego` | See recorded-results table below |

## Reduced cascadability check (related to paper Sec. 5.2.2)

RF-LEGO modules are designed to compose. `ae/scripts/cascade.py` compares
classical and RF-LEGO pipelines on the real-world mmWave AE benchmarks. Every
pipeline reports exact-bin Detection Rate (DR) at a fixed operating point in a
common **FAR at the `10^-3` order of magnitude** regime:

- range / Doppler : front-end FT → Detector   (classical: Bluestein FFT → CA-CFAR)
- angle           : Beamformer → Detector     (classical: LASSO → CA-CFAR)

All learned paths use per-profile min-max normalization to `[0,1]`. Range and
Doppler retain 4 training cells and 2 guard cells per side. Angle alone uses the
fixed experimental setting of 18 training cells and 10 guard cells per side.

Both paths are read out at fixed, retained scalar cutoffs that are constants of
the artifact. RF-LEGO Detector does not expose a CA-CFAR-style `P_FA` parameter;
its logit cutoff defines the binary readout but does not change the trained
Detector. Each target contributes only its exact ground-truth center bin;
adjacent bins are negatives. We describe the two paths as **FAR-regime aligned**
because both operate at the same `10^-3` order of magnitude. The paper does not
specify the numerical front-end-to-Detector conversion.

**Recorded results:**

| Pipeline | CA-CFAR train/guard per side | CA-CFAR DR | RF-LEGO DR |
|---|---:|---|---|
| range   | 4 / 2  | 0.780 | 0.880 |
| Doppler | 4 / 2  | 0.400 | 0.960 |
| angle   | 18 / 10 | 0.560 | 0.760 |

All three pipelines use strict exact-bin matching, at FAR of the `10^-3` order
of magnitude.

**Scope vs. the paper.** Both families of operating points sit in the same
`10^-3`-order low-FAR regime, so the two paths compare DR within that regime.
Training and angle inference share min-max normalization; the remaining mismatch
is between the compact synthetic Detector profile shapes and Beamformer/LASSO
spectra. These values must not be presented as an exact reproduction of paper
Sec. 5.2.2.
