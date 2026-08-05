> **Status (final).** This document was written at design time; its internal
> "Status: DESIGN + SKELETON ONLY" line below is superseded. The module *was*
> built and trained: held-out survival-prediction AUC rose from 0.500 to 0.9965,
> and on the validation scenes it improved every well-populated class
> (Forklift +2.16 AssA / +1.04 HOTA, PalletTruck +1.15 AssA, Person +0.05 AssA)
> while regressing the two classes with only two objects present
> (NovaCarter −0.71 HOTA, Transporter −0.10 HOTA). That violated our per-class
> no-regression criterion, so **it was not part of the rank-1 submission**.
> It is released as exploratory work. See
> [EXPERIMENTS.md](EXPERIMENTS.md#5-cvc-assoc-a-learned-survival-gate) for the
> outcome and caveats, and note that three of the 59 input features
> (age, time gap, speed) are zero-filled in the as-built head.
>
> Sections 7-8 below ("Deliverables", "Open questions for coordinator review")
> are design-process artifacts, kept for transparency about how the module was
> gated before it was allowed to consume GPU time.

---

# CVC-Assoc — Learned Cross-View Consensus Temporal-Survival Gate

**Our own architectural contribution** for AI City 2026 Track 1. A small, differentiable
per-temporal-instance **survival gate** `g_i ∈ (0,1)` that **replaces the scalar
`confidence_decay` (0.9)** and the hand-tuned post-hoc CCS decay inside the Sparse4D
instance bank. Trained on GT track IDs, backbone + detection **frozen**. Targets **AssA**
— the only 3D-HOTA sub-metric where we trail Team34 (58.35). Floor-safe by construction,
causal by construction.

Status: DESIGN + SKELETON ONLY. No training launched (awaiting coordinator review).

---

## 1. Where it plugs in (exact hooks)

Path: `.../cv/sparse4d/model/instance_bank.py`, method `cache()` (lines 460–504 in the
current tree). The temporal-survival multiply is here:

```python
# instance_bank.py :: cache()   (current, CCS-era)
cls_ids   = confidence.argmax(dim=-1)                    # [bs, num_anchor]  NGC-order class
confidence = confidence.max(dim=-1).values.sigmoid()     # [bs, num_anchor]  per-anchor score
if self.confidence is not None:                          # we have a previous frame
    if self.ccs_enabled:
        decay = self._ccs_decay_vec(anchor[:, :T], cls_ids[:, :T])   # [bs, T]  (geometry heuristic)
    else:
        decay = self.confidence_decay                    # scalar 0.9  (base path)
    confidence[:, :T] = torch.maximum(
        self.confidence * decay,                         # <-- SURVIVAL MULTIPLY (the hook)
        confidence[:, :T],
    )
self.temp_confidence = confidence
# ... topk(confidence, T, feature, anchor) -> self.cached_feature / self.cached_anchor
```

where `T = self.num_temp_instances` (the temporal slots), `self.confidence` is the
**previous** frame's cached per-slot score, and `confidence[:, :T]` is the **current**
detection score for the same slots.

**CVC-Assoc replaces `decay`** with a learned gate:

```python
    elif self.cvc_enabled:
        decay = self.cvc_head(                           # [bs, T] in (0,1)
            prev_conf = self.confidence,                 # [bs, T]
            cur_conf  = confidence[:, :T],               # [bs, T]
            anchor_t  = anchor[:, :T],                   # [bs, T, box_dims]
            feat_t    = self.cached_feature,             # [bs, T, embed_dims]  (identity)
            cvc_feat  = self._cvc_geom_feat(anchor[:, :T], cls_ids[:, :T]),  # [bs, T, G]
        )
```

The **downstream effect** (why this is not offline-replayable and must run in-model): the
modulated `temp_confidence` → `topk` → `cached_feature`/`cached_anchor` → next frame's
`temp_instance_feature`/`temp_anchor` → temporal cross-attention (`op == "temp_gnn"`,
`sparse4d_head.py:458–469`) → changes **all** subsequent-frame predictions. The gate is a
recurrent control signal, exactly like the scalar decay it replaces — so it trains/infers
inside the forward pass, not as a post-hoc pass over emitted boxes.

## 2. The gate module `CVCAssocHead`

A tiny MLP (frozen everything else → this is the only thing that trains). Per temporal
slot `i`, input feature vector `x_i`:

| block | components | dim |
|---|---|---|
| **survival state** | `prev_conf_i`, `cur_conf_i`, `logit(prev)`, `Δconf = cur−prev`, `age_i` (frames since first cached, normalized), `time_gap_i` (Δt to prev, from `metas.timestamp`) | 6 |
| **geometry / anchor** | decoded box: `X,Y,Z` (recentered), `logW,logL,logH`, `sin(yaw),cos(yaw)`, `‖velocity‖` if present | ~9 |
| **cross-view consensus (CVC)** | from `_cvc_geom_feat` (see §3): `c_geo` (fraction of in-FoV cams hitting an independent RT-DETR det), `n_vis` (cam count, normalized), `n_hit`, `mean_det_conf` over hits, `max_iou2d` of the projected box vs matched det, per-class one-hot of the predicted class (7) | ~12 |
| **identity** | `cached_feature_i` (embed_dims, e.g. 256) projected by a learned linear to 32-d | 32 |

`x_i` (≈ 59-d) → MLP `[59 → 128 → 128 → 1]` (GELU, LayerNorm) → scalar `s_i` →
`g_i = decay_lo + (decay_hi − decay_lo) · sigmoid(s_i + b0)`, with `decay_lo=0.80`,
`decay_hi=0.98`.

**Floor-safe init (critical).** Initialize the final layer weight ≈ 0 and bias `b0` so the
gate outputs the scalar base at init:
`g_i(init) ≈ 0.90 = decay_lo + (decay_hi−decay_lo)·sigmoid(b0)` → `sigmoid(b0)=0.5` →
`b0=0`, and set `decay_lo=0.80, decay_hi=1.00`→ midpoint 0.90. With near-zero final
weights, `g_i ≡ 0.90` at init = **literal base cd0.9**. Any improvement is learned on top
of the proven floor; a bad init cannot move us off 58.92.

**Runtime switch.** `instance_bank.enable_cvc(head, ...)` sets `self.cvc_enabled=True`;
absent that call, `cvc_enabled=False` → scalar `confidence_decay` path → byte-identical to
base (same guarantee the CCS scene-gate gave us). CVC and CCS are mutually exclusive
(assert not both enabled).

## 3. Cross-view consensus feature `_cvc_geom_feat`

**Reuses the validated CCS projection machinery** (`_ccs_decay_vec`, AUC 0.873 TP/FP
separation). Same inputs registered by `enable_cvc(cams, dets, shift, ngc2off, ...)`:
`cams = {cam_id: {K,R,t,W,H}}`, `dets = {"cam|frame": [[off_cls,x1,y1,x2,y2,conf],...]}`
from the **independent** RT-DETR detector (no circularity with Sparse4D).

For each temporal anchor, project center `Pc` and footprint `Pf=(X,Y,Z−H/2)` into every
camera; a camera *sees* the box iff the center projects in-frame; it *consents* iff (same
class) AND (center-uv OR footprint-uv ∈ a same-class det box). Emit per slot:
`c_geo = n_hit/max(n_vis,1)`, `n_vis`, `n_hit`, `mean_det_conf(hits)`, `max_iou2d`,
class one-hot. This is the **differentiable-input** feature (the projection itself is
`no_grad` geometry; gradients flow only through the MLP on top of these scalars, and
through the `cached_feature` identity branch).

**Variant B (optional, heavier):** instead of / in addition to RT-DETR consensus, pool the
**Sparse4D backbone feature maps** at the projected pixel across cameras (mean/max +
consenting-cam count). Gives a fully self-contained (no external detector) consensus
signal. Deferred to a second ablation — Variant A (RT-DETR) is primary because the signal
is already validated.

## 4. Supervision — (temporal instance → GT-track survival)

Train scenes W000–019 have `ground_truth.json` (3D boxes + persistent track IDs). Labeling
(script `cvc_label_targets.py`, §6):

1. Run the **frozen** Sparse4D over each train sequence (RGB-only, recentered, `no_temporal
   off` — we need the real temporal bank to populate slots). At each `cache()` call, dump
   for every temporal slot `i`: the feature vector `x_i` (§2), the slot's 3D anchor, its
   predicted class, and its instance-bank track id.
2. **Match** each slot's anchor to GT at that frame by 3D-IoU (≥ 0.3, per-class). 
3. **Target label** `y_i`:
   - `y_i = 1` (should survive, g→1) iff the slot matches a GT box whose **track id
     persists** ≥ K future frames (K=5) — a real, trackable object.
   - `y_i = 0` (should decay, g→0) iff the slot is **unmatched** (no GT within IoU), or
     matches a GT that **disappears** next frame (stale), or is a **duplicate** (a
     higher-confidence slot already claimed that GT track this frame).
4. Persist `(x_i, y_i, scene, frame, slot)` to a compact `.npz` per scene.

This turns a recurrent-control learning problem into **supervised BCE on cached vectors** —
the frozen forward is run once; the MLP then trains in minutes on the dumped vectors.

## 5. Loss & training

- **Primary:** `BCEWithLogits(s_i, y_i)`, class-balanced (positives rarer for rare
  classes) via `pos_weight` per NGC class.
- **Association-consistency (optional aux):** encourage monotonicity — among slots mapped
  to the *same* GT track in a frame, the surviving (kept-by-topk) one should have the
  highest `g`. Hinge on pairwise `g` differences. Small weight (0.1).
- **Floor anchor (regularizer):** `λ·(g_i − 0.90)²` with tiny `λ` so, absent strong
  evidence, the gate stays near base (defense-in-depth on top of the init).
- Optimizer AdamW lr 1e-3 (only the MLP trains, ~30k params), 5–20 epochs over the dumped
  vectors, early-stop on a held-out train scene. **No backbone/detector gradients.**

Because only the MLP trains on cached vectors, this needs **minutes on one GPU** (or CPU) —
it does not compete with the guarded-FT run.

## 6. Causal-safety argument (preserve the online +10% bonus)

Every input to `g_i` is available **at or before the current frame**:
`prev_conf`/`cached_feature` = strictly past (last frame's bank); `cur_conf`/`anchor` =
current frame; `cvc_feat` = current-frame RT-DETR dets on current images; `age`/`time_gap`
= past timestamps. **No future frame is read.** The gate is applied in `cache()` which runs
once per frame in temporal order. Prefix-invariance holds by the same argument as the
scalar decay (identical mechanism). A `verify_causal_cvc.py` check (deferred) will confirm
prediction for frame t is invariant to frames > t, exactly as done for the decay/hysteresis
ablations.

## 7. Deliverables in this task

- `CVC_ASSOC_DESIGN.md` — this note.
- `cvc_assoc_head.py` — the `CVCAssocHead` module skeleton + floor-safe init + the
  `enable_cvc`/`_cvc_geom_feat` integration stubs mirroring the CCS hooks.
- `cvc_label_targets.py` — the (instance → GT-track survival) labeling script that runs the
  frozen model over train pkls and dumps `(x_i, y_i)` vectors.

## 8. Open questions for coordinator review (before we spend GPU)

1. **Variant A (RT-DETR consensus) vs B (backbone-feature consensus) vs both** as the CVC
   feature. A is validated + external-independent (cleaner novelty claim); B is
   self-contained. Recommend A primary, B as ablation.
2. **decay range** `[0.80, 1.00]` (proposed) vs the CCS-tuned `[0.85, 0.93]`. Wider range =
   more expressive but more room to hurt; the floor-init protects either way.
3. Whether to **also** let the gate modulate `max_time_interval` (how long a slot may be
   retained) — currently only the per-frame decay is learned.
4. Run the labeling forward with the **base** ckpt or the **guarded-FT** ckpt (task 1) once
   it passes the val-gate — the FT ckpt may shift the slot statistics.

---

## 9. LOCKED DECISIONS + AS-BUILT (2026-07-10, supersedes §8)

The coordinator locked all four §8 questions; the code now reflects them. This section is
the **authoritative** description of what is implemented (it overrides the skeleton pseudo-
code in §1–3 where they differ).

### 9.1 Locked decisions
1. **Primary feature = Variant A** (RT-DETR cross-view consensus, validated AUC 0.873).
   Variant B (backbone-feature pooling) is a later ablation, **not** wired in v1.
2. **Decay range = [0.80, 0.99]** (`decay_lo=0.80, decay_hi=0.99`). Widened above the
   CCS-tuned [0.85, 0.93] so a high-consensus track can be held toward ~1.0 through
   occlusion — the lever the scalar could not reach. Capped at **0.99** (not 1.00) as the
   coordinator safety margin: even a maximally-confident carried track still decays slowly,
   and `max_time_interval` hard-caps ghosts.
3. **`max_time_interval` is FIXED, not learned** in v1 (one new variable at a time).
4. **Label with the BASE ckpt** (not the FT ckpt) → CVC-Assoc is a clean, FT-independent
   contribution that composes with any detector checkpoint.

### 9.2 As-built feature vector (in_dim = 59, verified)
`build_feature()` in `cvc_assoc_head.py` assembles, per temporal slot:
- **state (6):** `prev_conf`, `cur_conf`, `logit(prev)`, `Δconf`, `age`(0/TODO), `time_gap`(0/TODO).
- **geom (9):** `X, Y, Z, logW, logL, logH, sin_yaw, cos_yaw, ‖vel‖`(0/TODO) — RECENTERED-world anchor.
- **cvc (5 + 7 one-hot = 12):** the 5 consensus scalars from `cvc_geom_scalars()` =
  `[c_geo, n_vis/ncam, n_hit/ncam, mean_matched_det_conf, max_matched_det_conf]`, plus the
  predicted-class one-hot (7). (Note: the skeleton §2 listed `max_iou2d`; the as-built 5th
  scalar is `max_matched_det_conf` — more directly tied to the RT-DETR evidence strength.)
- **id (32):** `id_proj(instance_feature_i)` — a **FROZEN random linear** 256→32 (see 9.4).

`age`, `time_gap`, `‖vel‖` are wired as zeros in v1 (marked `TODO(cvc)`); they are additive
inputs that default to a no-op and can be filled without changing the interface.

### 9.3 As-built feature timing (train == inference; the correctness fix)
The gate is computed in `cache()` **before** the topk selection. Both the offline labeler
(`CacheProbe`) and the runtime (`InstanceBank._cvc_decay_vec`) now use the **current-frame,
pre-topk** temporal slots, all indexed `[:, :T]` on the *same* tensors:
- `feat_t   = instance_feature[:, :T]` (current refined id feature — **not** the post-topk
  `cached_feature`, which is misaligned with `anchor[:, :T]`; this was fixed in the labeler),
- `anchor_t = anchor[:, :T]`, `cur_conf = confidence[:, :T]`, `cls_ids = cls_ids[:, :T]`,
- `prev_conf = self.confidence` (previous frame's cached per-slot score, `[bs, T]`).

This guarantees the head is trained on exactly the vectors it sees at inference.

### 9.4 Shared frozen `id_proj` (why one init head is mandatory)
`build_feature` bakes `id_proj(feat_t)` into the dumped `X`. In v1 `id_proj` is a **frozen
random projection**; therefore the *same* `id_proj` weights must be used at (a) every scene's
labeling pass and (b) inference, or the 59-d vectors are mutually inconsistent. Enforced by a
single persisted init head:
```
python cvc_train_head.py make-init --out cvc_head_init.pt        # canonical head (seed=0)
python cvc_label_targets.py ... --head_init cvc_head_init.pt ... # every scene uses it
python cvc_train_head.py train --head_init cvc_head_init.pt --npz cvc_labels/*.npz \
       --out cvc_head_trained.pt                                 # freezes id_proj, trains MLP
```
(Trainable `id_proj` would require dumping the raw 256-d `feat_t` — a v2 ablation.)

### 9.5 As-built runtime integration (additive, floor-safe)
`instance_bank.py` gained, guarded by `getattr(self, "cvc_enabled", False)` (defaults False):
- `__init__`: CVC attrs all disabled (`cvc_enabled=False`, `cvc_head=None`, …).
- `set_cvc_frame(frame_id)` — mirror of `set_ccs_frame`.
- `_cvc_decay_vec(feat_t, anchor_t, cur_conf, cls_ids_t)` — computes `cvc_geom_scalars` per
  batch item, calls `head.build_feature` then `head(...)`, returns `g ∈ (0.80, 0.99)`.
- `cache()`: a **new first branch** `if getattr(self,'cvc_enabled',False): decay=_cvc_decay_vec(...)`
  above the existing `elif self.ccs_enabled` / `else scalar` branches. CVC and CCS are mutually
  exclusive (asserted in `enable_cvc`). Disabled → the exact pre-existing code path runs, so
  the byte-for-byte base/CCS behavior (and the armed ccs2c submission) is untouched.

`run_infer.py` gained flag-gated `--cvc_head/--cvc_dets/--cvc_calib/--cvc_conf` (default off →
byte-identical). When set, it loads the trained head, parses calibration exactly as the CCS
path, calls `enable_cvc`, and `set_cvc_frame` each step. `sys.path` now includes `work/` so
`cvc_assoc_head`/`cvc_train_head` resolve in-process.

### 9.6 Floor-safety, verified
- `make-init` asserts `g_init == base_decay` (measured **0.900000**, in_dim=59).
- Synthetic separable train: init AUC 0.500 → 0.96; gate stayed in **[0.805, 0.986]**;
  `g(pos)=0.959 > g(neg)=0.832` (persistent tracks decay slowly, stale ones fast). ✔
- Disabled path is byte-identical (getattr guard + mutual exclusion with CCS).

### 9.7 Remaining before it consumes GPU (gated on codex review + FT freeing GPU0)
- Cache **RT-DETR train-scene dets** for the ft8 scenes (W000/002/004/006/010/013/014/016) —
  reuse `ccs_step1_dets.py` machinery pointed at the train split.
- Run `cvc_label_targets.py` (BASE ckpt, `--head_init cvc_head_init.pt`, `--dets/--calib`),
  then `cvc_train_head.py train`, then a **val-gate** on W020/021/022 via `run_infer.py
  --cvc_head …` → HOTA-eval vs zero-shot (bar: ties-or-beats zero-shot on **every** class).

