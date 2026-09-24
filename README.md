# STEAD-VLM

**A resource-aware multi-stage vision-language architecture with distilled explainable AI for
retail crime detection on edge computing devices** - code accompanying the M.Eng. thesis by
Nora Ehiomogue (Department of Computer Engineering, University of Benin, 2026).

STEAD-VLM is a three-stage cascade that spends compute only where it is needed:

```
CCTV frames (30 FPS)
  -> keep every 6th frame (5 FPS), cut into 16-frame clips
  -> Stage 0  YOLOv8-nano        drop clips with no person / relevant object
  -> Stage 1  X3D-L + STEAD-Fast anomaly probability; escalate if score > theta (0.65)
  -> Stage 2  Grad-CAM + MobileVLM V2 (LoRA)   evidence heat-map + natural-language explanation
  -> Telegram alert (text + Grad-CAM image)
```

Stage 2 is explanatory only: it never changes the Stage-1 score or decision.

> **Edge claim, stated precisely.** Efficiency figures in the thesis were measured in a Google
> Colab Pro session (NVIDIA T4) configured to *simulate* the memory / latency limits of low-cost
> edge hardware. No physical edge device was used, so they show that the workload fits those
> limits, not how it behaves on a Raspberry Pi or Jetson.

## Reported results (thesis Chapter 4, 1,260-clip held-out test set)

| System | Accuracy | Precision | Recall | F1 | AUC-ROC | Peak VRAM |
|---|---|---|---|---|---|---|
| STEAD-VLM | 94.13 | 90.72 | 95.04 | 92.83 | 93.12 | 3.2 GB |
| Cerberus | 96.50 | 94.90 | 96.43 | 95.66 | 97.20 | 9.1 GB |
| LAVAD | 84.20 | 78.82 | 82.74 | 80.73 | 86.74 | 8.8 GB |

The cascade is not the most accurate system (Cerberus is), but uses about 64 % less memory.
Throughput / latency are reported in the thesis as 28.4 FPS / 35.2 ms per frame; see
`stead_vlm/metrics.py::latency_summary` for exactly what those units count.
The local CCTV data, labels and captions are **not** released (privacy / store agreements).

## Repository layout

```
stead_vlm/        library code   (config, preprocessing, stage0, stead_fast, gradcam, vlm,
                                  explain, pipeline, alerts, metrics, data, overlay)
scripts/          prepare_data.py  extract_features.py  train_stead_fast.py  evaluate.py  demo.py
tests/            unit tests (no GPU, weights or data needed)
```

## Install

```bash
pip install -r requirements.txt
git clone https://github.com/agao8/STEAD third_party/STEAD                 # STEAD-Fast head (MIT)
git clone https://github.com/Meituan-AutoML/MobileVLM third_party/MobileVLM  # MobileVLM V2 (Apache-2.0)
export PYTHONPATH=$PWD/third_party/MobileVLM:$PYTHONPATH
```

The two upstream projects pin different PyTorch versions (STEAD 2.6, MobileVLM 2.0.1). This
repository has not been run against either pin - choose a version that satisfies both, or run
Stage 1 and Stage 2 in separate environments.

## Use

```bash
# unit tests (CPU only)
pytest -q

# run the cascade on a video; Telegram credentials only via environment variables
export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
python scripts/demo.py --video clip.mp4 \
    --stead-repo third_party/STEAD --stead-ckpt weights/stead_fast_head.pt \
    --vlm-base weights/MobileVLM_V2-1.7B --vlm-adapter weights/mobilevlm_lora --telegram

# evaluation on your own held-out split
python scripts/evaluate.py run   --test-csv test_split.csv --out results/cascade.csv \
    --stead-repo third_party/STEAD --stead-ckpt weights/stead_fast_head.pt
python scripts/evaluate.py score STEAD-VLM=results/cascade.csv Baseline=results/baseline.csv
```

Data preparation for your own footage: `python scripts/prepare_data.py --help`
(audit -> blur -> clips -> label -> npy -> manifest -> split -> augment).

## Status of each part

| Part | Status |
|---|---|
| Pre-processing, Stage-0 rules, clip sampling, data split / augmentation, explanation parsing, alerts, metrics, cascade logic | unit-tested (`pytest`) |
| X3D-L + STEAD-Fast wrapper, Grad-CAM, MobileVLM explainer, training / feature scripts | written against the official STEAD and MobileVLM code; **not executed** here (needs GPU, weights, data) |
| `scripts/train_stead_fast.py` | reference reconstruction - see its docstring |
| MobileVLM LoRA training | not included |

## Licence

Copyright (C) 2026 Nora Ehiomogue. Released under the **GNU Affero General Public License v3.0**
(see `LICENSE`). Stage 0 uses Ultralytics YOLOv8, which is itself AGPL-3.0, so this project is
released under the same licence.

Third-party components keep their own terms: Ultralytics YOLOv8 (AGPL-3.0; the off-the-shelf
COCO `yolov8n.pt` weights are downloaded by Ultralytics, not redistributed here), STEAD (MIT),
MobileVLM V2 (Apache-2.0 code; check the model weights and the CLIP vision tower), X3D /
PyTorchVideo (Apache-2.0; Kinetics-pretrained weights), MediaPipe (Apache-2.0). UCF-Crime is used
under its own research terms and is not redistributed.

## Status

Research code accompanying a thesis. The unit tests pass, but the GPU stages have not been
independently re-run from this repository; see "Status of each part" above.
