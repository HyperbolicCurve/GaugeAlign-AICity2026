# Data

## Dataset

**`nvidia/PhysicalAI-SmartSpaces`**, subset **`MTMC_Tracking_2026`** — the AI City Challenge 2026
Track 1 benchmark. License: **CC-BY-4.0**.

- Hugging Face: <https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces>
- Challenge: <https://www.aicitychallenge.org/>

Seven classes, ids fixed by the official protocol:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| | Person | Forklift | NovaCarter | Transporter | FourierGR1T2 | AgilityDigit | PalletTruck |

## You only need the videos

The dataset ships RGB video **and** per-camera depth (`.h5`), and the depth archive is ~2.4 TB.
**GaugeAlign never touches depth.** Inference is RGB-only (the track requires it), and
`build_pkl.py` reads exactly two things per scene: `videos/Camera_XXXX.mp4` and
`calibration.json`. `ground_truth.json` is needed only to *score* the validation studies.

That makes reproduction far cheaper than the full download suggests:

| what you want to reproduce | download | size |
|:--|:--|--:|
| the rank-1 submission | `test/` videos + calibration | **~6.2 GB** |
| the gauge stress curve / factorial | `val/` videos + calibration + GT | **~4.2 GB** |
| everything in the paper | both of the above | **~11 GB** |
| *(depth, `.h5`)* | *not used* | *~2.4 TB* |

## Layout

`DATASET_ROOT` must point at the directory holding the splits:

```
MTMC_Tracking_2026/
├── train/
├── val/
│   ├── Warehouse_020/           16 cameras
│   ├── Warehouse_021/           16 cameras
│   └── Warehouse_022/            4 cameras   <- the scene that collapses to 0
│       ├── calibration.json          intrinsics + extrinsics (this is what C-bar comes from)
│       ├── ground_truth.json         val/train only
│       ├── map.png
│       └── videos/
│           ├── Camera_0000.mp4
│           └── ...
└── test/
    ├── Warehouse_023/           20 cameras, 9000 frames
    ├── Warehouse_024/           20 cameras, 9000 frames
    ├── Warehouse_025/           10 cameras, 9000 frames
    ├── Warehouse_026/            4 cameras, 1800 frames
    └── Warehouse_027/            7 cameras, 1800 frames
```

```bash
export DATASET_ROOT=/path/to/MTMC_Tracking_2026
```

## Downloading

```bash
pip install "huggingface_hub[cli]"

# test videos + calibration only (enough for the rank-1 submission)
hf download nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
    --include "MTMC_Tracking_2026/test/**" \
    --exclude "**/depth*" "**/*.h5" \
    --local-dir /path/to/data

# add val for the controlled studies
hf download nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
    --include "MTMC_Tracking_2026/val/**" \
    --exclude "**/depth*" "**/*.h5" \
    --local-dir /path/to/data
```

The dataset is gated: accept the terms on the Hugging Face page and `hf auth login` first. Behind
a slow link, `HF_ENDPOINT=https://hf-mirror.com` works.

## Calibration is the only input the method needs

Everything GaugeAlign does at the geometry level comes out of `calibration.json`. Each sensor
carries a 3×4 `extrinsicMatrix` (world→camera) and a 3×3 `intrinsicMatrix`; the camera centre is
$C_i = -R_i^\top t_i$ and the canonical shift is their mean. Check any scene without a model,
a GPU, or ground truth:

```bash
python gaugealign/scene_center.py --scene_dir $DATASET_ROOT/val/Warehouse_022 \
                                  --anchor $NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy
```

```
  camera xy extent: x[-83.4, -43.0]  y[-95.0, -71.2]
  C-bar (xy) = (-62.7303, -83.0016)   |C-bar| = 104.04 m
  anchor prior (900 anchors, training gauge): mean xy = (4.20, 0.20)  ...
  -> in the absolute frame the scene sits 104.0 m from where the anchors live;
     canonicalization removes exactly this offset.
```

Pinned values for every scene are in [`configs/scenes.yaml`](../configs/scenes.yaml).

## Disk

Decoded frames dominate, not the dataset. `build_pkl.py --half_res` writes 960×540 JPEGs
(q=92) — about 7× smaller than full resolution and, because the model resizes by 0.5 anyway,
**identical model input**.

Rough per-scene peak: `frames ≈ n_cameras × n_frames × ~60 KB`. Warehouse_023 (20 cameras ×
9000 frames) is the worst at ~11 GB. `reproduce_cd95.sh` deletes each scene's frames as soon as
inference finishes, so peak usage is one or two scenes at a time; budget **~30 GB** of scratch.
Pass `KEEP_FRAMES=1` to retain them for repeated runs.

## Submission format

11 space-separated fields per line, floats to 2 decimals, zipped as `track1.txt` inside
`track1.zip` (≤ 50 MB):

```
<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
```

`gaugealign/make_submission.py` enforces this: it validates every row, drops malformed ones with
a report, renumbers object ids to be positive and dense per (scene, class), and checks the size
limit.
