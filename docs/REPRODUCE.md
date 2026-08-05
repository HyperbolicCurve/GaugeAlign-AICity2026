# Reproducing the rank-1 submission

Target: **3D HOTA 56.5447** (DetA 55.6444 / AssA 49.3929 / LocA 79.5558) on the final full hidden
test set — 1st of 10 teams, AI City Challenge 2026 Track 1.

Machine-readable form of everything below: [`configs/gaugealign_cd95.yaml`](../configs/gaugealign_cd95.yaml).

## One command

```bash
export DATASET_ROOT=/path/to/MTMC_Tracking_2026
export NGC_MODEL_DIR=/path/to/sparse4d_rn50_vtrainable_v2.2
bash scripts/reproduce_cd95.sh
```

Output: `work/track1.zip`. On 2× RTX 3090 the full five scenes take roughly a day, dominated by
video decode.

Knobs (all optional): `WORK_DIR`, `GPUS` (default `0,1`), `PREP_ENV`, `INFER_ENV`, `SCENES`,
`KEEP_FRAMES`.

Start small — Warehouse_026 is 4 cameras × 1800 frames and finishes in well under an hour:

```bash
SCENES=26 GPUS=0 bash scripts/reproduce_cd95.sh
```

## What the pipeline does

Three stages per scene. Only stage 2 involves the network, and its weights never change.

### 1. Canonicalize the gauge — `build_pkl.py --auto_center`

Computes $\bar{C} = \frac{1}{N}\sum_i(-R_i^\top t_i)$ from `calibration.json` and rewrites every
extrinsic translation as

$$t_i' = t_i + R_i\bar{C}$$

so the scene is expressed in the camera-centroid frame — the gauge the anchor prior was trained
in. Images are untouched. The shift is stored in the pkl metadata so it can be inverted exactly.
The same pass decodes frames to 960×540 (`--half_res`; the model halves the resolution anyway, so
this is the *identical* model input at ~7× less disk).

Confirm it is active — the log must show a non-zero shift:

```
world re-center shift C = [-6.8, -1.71] (predictions add this back)
```

### 2. Frozen detector + inverse — `run_infer.py`

Runs Sparse4D unchanged, then per box:

- **gauge inverse**: adds $\bar{C}$ back to the centre, returning it to the site frame. Round-trip
  error < 10⁻⁶ m, so scoring is invariant to the choice of frame and the operator cannot leak
  target information.
- **protocol alignment** (`--remap`): the model's class indices are not the official Track-1 ids.
  $0{\to}0,\ 1{\to}4,\ 2{\to}5,\ 3{\to}2,\ 4{\to}3,\ 5{\to}1,\ 6{\to}6$.
- **temporal prior** (`--conf_decay 0.95`): the instance bank multiplies an unmatched instance's
  confidence by $\lambda$ each frame. The released $\lambda = 0.8$ is a prior on occlusion length
  tuned to the training distribution; 0.95 stretches the survival horizon by
  $\ln 0.8/\ln 0.95 \approx 4.4\times$.

### 3. Floor-ROI filter + package

`floor_roi.py` drops boxes whose xy leaves the camera-centre bounds by more than 20 m, or whose z
is outside [−0.5, 3.2] m. `make_submission.py` validates and zips.

**No temporal post-processing is applied.** Gap interpolation and NMS were tried
(`bundlefix`, 56.38) and discarded: gap interpolation reads future frames and forfeits the online
bonus. See [the causality test](#verify-the-online-property).

## Exact hyper-parameters

| | value | where |
|:--|:--|:--|
| checkpoint | `sparse4d_warehouse_v2.2_r50.pth`, `deployable_v2.2` | NGC, **frozen** |
| anchors | `_ov_kmeans900_v2.2_r50.npy` (900) | NGC, unmodified |
| canonical shift | $\bar{C}$ per scene, from calibration | `--auto_center` |
| class remap | 0→0, 1→4, 2→5, 3→2, 4→3, 5→1, 6→6 | `--remap` |
| colour order | BGR (`--to_rgb 0`) | matches training |
| confidence decay $\lambda$ | **0.95** | `--conf_decay 0.95` |
| score threshold | 0.2 | `--score_thr 0.2` |
| decoder threshold | 0.0 | `--dec_thr 0.0` |
| resolution | 960×540 | `--half_res` |
| frame stride | 1 (native rate) | |
| floor ROI | margin 20 m, z ∈ [−0.5, 3.2] | `floor_roi.py` |
| temporal post-proc | none | — |

## Per-scene shift values

Derived from calibration alone, so they are auditable without running anything:

| scene | cameras | frames | shift_x | shift_y | \|C̄\| |
|:--|--:|--:|--:|--:|--:|
| Warehouse_023 | 20 | 9000 | −43.5309 | −50.9003 | 66.98 |
| Warehouse_024 | 20 | 9000 | −43.5309 | −50.9003 | 66.98 |
| Warehouse_025 | 10 | 9000 | −11.3245 | 2.9237 | 11.70 |
| Warehouse_026 | 4 | 1800 | −6.8022 | −1.7110 | 7.01 |
| Warehouse_027 | 7 | 1800 | −3.7758 | 4.5543 | 5.92 |

Regenerate:

```bash
python gaugealign/scene_center.py --dataset_root $DATASET_ROOT --split test --quiet
```

`--auto_center` computes these at runtime; the table exists so a run can be checked against it.
To pin them explicitly, pass `--shift_x/--shift_y` instead.

## Verify the online property

```bash
python scripts/prefix_check.py work/pred_test_cd95_26.txt
```

Post-processing a truncated sequence must give byte-identical output to truncating the
post-processed full sequence. The `gapfill` stage is included **expecting it to fail** — it is the
non-causal variant we discarded, and it is what proves the test discriminates.

```
  submitted    causal                           as expected
  hysteresis   causal                           as expected
  gapfill      reads future frames (16/200)     as expected
```

Recorded run: [`results/prefix_check_warehouse026.txt`](../results/prefix_check_warehouse026.txt).

Note the input must be the 12-column prediction (with scores); the packaged `track1.txt` has that
column stripped.

## Scoring

Hidden-test labels are not public — the numbers above come from the challenge server, which we
cannot reproduce offline. For validation scenes:

```bash
python scripts/evaluate_hota_present.py \
    --gt  work/gt_22.txt \
    --pred work/pred_22.txt
```

`evaluate_hota_present.py` implements the official aggregation (per-scene mean over the classes
**present** in that scene's GT, weighted across scenes by object count). Plain
`evaluate_hota.py` averages over all 7 classes and scores absent ones as 0, which heavily
penalizes partial scenes such as Warehouse_022 — use it only for like-for-like comparisons.

Both are development approximations, not the official server.

## Expected deviations

- **Frame decode.** OpenCV seeking is not bit-exact across versions/builds. Different JPEG output
  shifts scores by well under 0.1 HOTA. It will not move 56.5 by anything visible.
- **λ = 0.9 vs 0.95.** These are within split noise and the ordering *flips* between splits: on
  the 50% development split 0.9 won (58.92 vs 58.73); on the full set 0.95 won (56.54 vs 56.37).
  Do not read a 0.1 HOTA difference as signal.
- **GPU count.** Affects only wall-clock; scenes are independent and results are deterministic
  per scene.
