# RF-LEGO Artifact Evaluation

This repository contains the artifact-evaluation package for RF-LEGO. It provides
a one-click workflow for evaluating modality-specific module results from the
prepared real-world `.npz` files under `ae_data/` for:

- Frequency Transform
- Beamformer
- Detector

It also provides a compact **cascadability** check related to paper Sec. 5.2.2
via `ae/scripts/cascade.py`: range/Doppler use FT→Detector and angle uses
Beamformer→Detector. This reduced AE check does not reproduce the paper's full
data and baseline coverage.

Current AE data files:

| File | Modality | Module/task |
|---|---|---|
| `mmwave_beamformer.npz` | mmWave | Beamformer |
| `mmwave_dopplerft.npz` | mmWave | Doppler FT |
| `mmwave_rangeft.npz` | mmWave | Range FT |
| `uwb_detector.npz` | UWB | Detector |
| `uwb_dopplerft.npz` | UWB | Doppler FT |
| `wifi_dopplerft.npz` | WiFi | Doppler FT |

Project metadata:

- `uv.lock` records the package-resolution lockfile used for this artifact.
- `CITATION.cff` provides citation metadata; replace anonymous placeholders with final public metadata before release.

Start here:

- One-click Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anonymous-rflego-ae/mobicom26_ae/blob/main/ae/RF_LEGO_AE_Colab.ipynb)
- Colab notebook file: `ae/RF_LEGO_AE_Colab.ipynb`
- AE instructions: `ae/README.md`
- AE benchmark files: `ae_data/*.npz`
- Pretrained weights: `ae/results/weights/`

The Colab notebook runs installation, module smoke tests, evaluation, plotting,
the cascadability experiment, and result summary generation with the shipped
pretrained weights under `ae/results/weights/`.

Local workflow:

```bash
pip install -e .
pip install -r ae/requirements.txt
python ae/scripts/evaluate.py --module all
python ae/scripts/plot_reproduction.py
python ae/scripts/cascade.py
```

Reviewers do not need to train anything: the whole workflow runs on the shipped
pretrained weights under `ae/results/weights/`.

Before public release, replace the anonymous GitHub owner/repository and author
placeholders in `pyproject.toml`, `README.md`, `ae/README.md`,
`ae/RF_LEGO_AE_Colab.ipynb`, and `CITATION.cff`, then
publish the Zenodo record and add its DOI to the final camera-ready metadata.
