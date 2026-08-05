# Installation

Two conda environments, because they have genuinely different CUDA needs:

| env | purpose | torch | CUDA |
|:--|:--|:--|:--|
| `gaugealign-prep` | canonicalization, video decode, scoring, packaging | 2.5.1 | 12.1 |
| `gaugealign` | Sparse4D inference (needs the CUDA extension) | 2.0.1 | 11.8 |

You can collapse them into one if your toolchain builds the deformable-aggregation extension
against a newer CUDA — nothing in the method depends on the split. The scripts read the env names
from `PREP_ENV` and `INFER_ENV`, so a single-env setup is just
`PREP_ENV=gaugealign INFER_ENV=gaugealign bash scripts/reproduce_cd95.sh`.

Reference hardware: 2× RTX 3090 (24 GB). Inference needs ~10 GB per scene; disk for decoded
frames is the real constraint (see [DATA.md](DATA.md#disk)).

## 1. Environments

```bash
# --- data prep / evaluation -------------------------------------------------
conda create -n gaugealign-prep python=3.10 -y
conda activate gaugealign-prep
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.2.6 opencv-python==4.12.0.88 scipy==1.15.3 \
            h5py==3.16.0 pyyaml==6.0.2 tqdm==4.67.1 pillow==11.3.0 matplotlib

# --- Sparse4D inference -----------------------------------------------------
conda create -n gaugealign python=3.10 -y
conda activate gaugealign
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 omegaconf==2.3.1 opencv-python==4.9.0.80 scipy==1.15.3 \
            pyyaml==6.0.3 tqdm==4.68.4 einops==0.8.2 timm==1.0.27 pillow
```

The inference env needs a matching CUDA toolkit (`nvcc`) to build the extension in step 2 —
CUDA 11.8 for torch 2.0.1. Check with `nvcc --version`.

Video decoding uses OpenCV's `VideoCapture`, so the `opencv-python` wheel (which bundles FFmpeg)
is sufficient; no separate FFmpeg install is required.

## 2. NVIDIA TAO backend + our patch

We do **not** vendor NVIDIA's code. The upstream repository is Apache-2.0 and this repo carries
only the diff, so it stays obvious which lines are ours.

```bash
bash scripts/setup_tao_backend.sh
```

That clones [`NVIDIA/tao_pytorch_backend`](https://github.com/NVIDIA/tao_pytorch_backend) at
`Release/7.0.1` (`d99ff772e2cff894015d81a96d2cb92119000e19`) into `third_party/` and applies
`patches/tao_sparse4d_gaugealign.patch`. Then build the CUDA extension:

```bash
conda activate gaugealign
cd third_party/tao_pytorch_backend/nvidia_tao_pytorch/cv/sparse4d/model/ops
python setup_ext.py build_ext --inplace
cd -
```

Verify:

```bash
python -c "import sys; sys.path.insert(0,'third_party/tao_pytorch_backend'); \
           from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import Sparse4D; print('ok')"
```

### What the patch changes

Five files, all justified:

| file | change | needed for the rank-1 run? |
|:--|:--|:--|
| `backbone_v2/__init__.py` | import backbones lazily, so missing optional deps (`transformers`, `open_clip`, …) don't break a ResNet-50 model | **yes** — import fix |
| `sparse4d_head.py` | plumb `max_time_interval` through to the instance bank | **yes** — 1 line |
| `ops/setup_ext.py` | *new*: build the deformable-aggregation CUDA extension | **yes** — build helper |
| `instance_bank.py` | our per-instance survival gate (CVC-Assoc) and cross-view consensus hooks | no — **off by default** |
| `criterion.py` | optional background-query loss term (fine-tuning study only) | no |

The two research additions are disabled by default and gated behind explicit `enable_*()` calls,
so an unmodified run is bit-identical to upstream. The rank-1 submission uses none of them.

If you already have the backend elsewhere, skip the script and set `TAO_BACKEND=/path/to/it`.

## 3. Model weights

Download the **frozen** public checkpoint from the NVIDIA NGC catalog —
[`nvidia/tao/sparse4d_rn50`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/sparse4d_rn50),
version `deployable_v2.2`, under the NVIDIA Open Model License. `NGC_MODEL_DIR` must contain:

```
sparse4d_warehouse_v2.2_r50.pth     # ~575 MB   the detector, used frozen
_ov_kmeans900_v2.2_r50.npy          # ~78 KB    the 900 k-means anchors (the spatial prior)
experiment.yaml                     # ~8 KB     model/dataloader config
```

```bash
export NGC_MODEL_DIR=/path/to/sparse4d_rn50_vtrainable_v2.2
```

> The anchor `.npy` is the object this whole paper is about: 900 box centres stored in the
> **training** world frame. `python gaugealign/scene_center.py --scene_dir <scene> --anchor
> $NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy` prints where those anchors live versus where your
> scene actually is.

Weights are **not** redistributed here — get them from NGC under their license.

## 4. Local evaluation (optional)

Only needed to re-run the validation studies. Hidden-test numbers in the paper come from the
challenge server, which we cannot redistribute.

```bash
bash scripts/setup_trackeval.sh
python scripts/evaluate_hota.py --help
```

## 5. Data

See [DATA.md](DATA.md). Then go to [REPRODUCE.md](REPRODUCE.md).

## Troubleshooting

**`ImportError: deformable_aggregation_ext`** — the CUDA extension was not built, or was built in
the wrong env. Rebuild it (step 2) with `gaugealign` active and `nvcc --version` matching torch's
CUDA.

**`TAO PyTorch backend not found`** — run `scripts/setup_tao_backend.sh`, or export `TAO_BACKEND`.

**Zero detections everywhere** — this is precisely the failure the paper is about. Check that
canonicalization is on: the `build_pkl.py` log must print a non-zero
`world re-center shift C = [...]`. Without `--auto_center` (or an explicit `--shift_x/--shift_y`)
you are running the absolute-frame baseline, which scores ~0 on few-camera scenes.

**Patch fails to apply** — you are not on `Release/7.0.1`. `git -C third_party/tao_pytorch_backend
checkout d99ff772e2cff894015d81a96d2cb92119000e19` and retry.
