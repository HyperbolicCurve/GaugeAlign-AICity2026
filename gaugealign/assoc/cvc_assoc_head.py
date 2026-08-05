"""CVC-Assoc — learned cross-view-consensus temporal-survival gate for Sparse4D.

Our own architectural contribution (AI City 2026 Track 1). A small MLP that produces a
per-temporal-instance survival gate g_i in (decay_lo, decay_hi) and REPLACES the scalar
`confidence_decay` (0.9) / hand-tuned CCS decay inside InstanceBank.cache().

DECISIONS LOCKED (coordinator, 2026-07-10) — v1:
  1. PRIMARY FEATURE = Variant A: RT-DETR cross-view consensus (validated AUC 0.873).
     `cvc_geom_scalars()` below implements it (reuses the CCS projection). Variant B
     (backbone-feature pooling) is a later ablation, not wired in v1.
  2. DECAY RANGE = [0.80, 0.99]. Widened above the CCS-tuned [0.85,0.93] so the gate can
     push high-consensus tracks toward ~1.0 (hold through occlusion) — the lever the scalar
     couldn't reach. Capped at 0.99 (not 1.00) as the coordinator-optional safety margin, so
     even a maximally-confident carried track still decays slowly; max_time_interval hard-caps
     ghosts and the val-gate catches regressions.
  3. max_time_interval is FIXED (not learned) in v1 — one new variable at a time.
  4. LABEL with the BASE ckpt (not FT) — keeps CVC-Assoc a clean, FT-independent contribution.

Floor-safe by construction: at init g_i == 0.90 == base cd0.9; a runtime switch (`enable_cvc`
not called) => scalar path => byte-identical to base. Causal: reads only current+past frames.
See docs/CVC_ASSOC_DESIGN.md for the full design + causal-safety argument.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# NGC-order class names (7-class taxonomy); index = predicted cls id used by the bank.
NUM_CLASSES = 7


class CVCAssocHead(nn.Module):
    """Per-temporal-slot survival gate. Only module that trains (backbone+detector frozen).

    Input per slot i is assembled by `build_feature(...)` into x_i of dim `in_dim`; the MLP
    maps x_i -> scalar logit s_i; the gate is
        g_i = base_decay + amp * tanh(s_i),   amp = min(base-lo, hi-base).
    BIT-EXACT floor-safe init: the final layer's weight AND bias are zero -> s_i == 0 ->
    tanh(0) == 0 -> g_i == float32(base_decay) bit-for-bit, identical to the scalar
    `confidence_decay` path, so a top-k tie can never diverge from the proven floor. (The old
    `lo + (hi-lo)*sigmoid(b0)` form was only ~1e-8 off base and could flip a top-k tie.)
    """

    def __init__(self, embed_dims=256, id_proj_dim=32, hidden=128,
                 decay_lo=0.80, decay_hi=0.99, base_decay=0.90):
        super().__init__()
        self.decay_lo = float(decay_lo)
        self.decay_hi = float(decay_hi)
        self.base_decay = float(base_decay)
        # id (instance feature) compressor
        self.id_proj = nn.Linear(embed_dims, id_proj_dim)
        # feature layout dims (keep in sync with build_feature)
        self.d_state = 6          # prev_conf, cur_conf, logit(prev), dconf, age, time_gap
        self.d_geom = 9           # X,Y,Z, logW,logL,logH, sin,cos yaw, |vel|
        # cvc scalars (Variant A, from cvc_geom_scalars):
        #   [c_geo, n_vis/ncam, n_hit/ncam, mean_matched_det_conf, max_matched_det_conf]
        self.d_cvc = 5 + NUM_CLASSES   # 5 consensus scalars + cls onehot
        self.d_id = id_proj_dim
        self.in_dim = self.d_state + self.d_geom + self.d_cvc + self.d_id
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # ---- BIT-EXACT floor-safe init: g_i == base_decay (float32) at init ----
        # amp is symmetric about base and bounded by BOTH requested bounds; the 0.99 hi cap is the
        # binding constraint (amp=0.09 for [0.80,0.99] about 0.90 -> effective range [0.81,0.99]).
        # The narrower low end (0.81 vs 0.80) is immaterial for stale-track suppression; the high
        # end (holding good tracks toward 0.99 through occlusion) is preserved exactly.
        self.amp = float(min(base_decay - decay_lo, decay_hi - base_decay))
        self.eff_lo = float(base_decay - self.amp)
        self.eff_hi = float(base_decay + self.amp)
        nn.init.zeros_(self.mlp[-1].weight)          # s == 0 at init (no feature contribution)
        nn.init.zeros_(self.mlp[-1].bias)            # s == 0 at init => tanh(0)=0 => g==base
        # float32 constants so forward is bit-exact against the scalar-decay path
        self.register_buffer("_base", torch.tensor(float(base_decay), dtype=torch.float32))
        self.register_buffer("_amp", torch.tensor(self.amp, dtype=torch.float32))

    def forward(self, feat):
        """feat: [bs, T, in_dim] assembled by build_feature. Returns g: [bs, T] in
        (base-amp, base+amp). At init s==0 -> g == float32(base_decay) exactly."""
        s = self.mlp(feat).squeeze(-1)               # [bs, T] logit (BCE-trained)
        g = self._base.to(feat.dtype) + self._amp.to(feat.dtype) * torch.tanh(s)
        return g

    def build_feature(self, prev_conf, cur_conf, anchor_t, feat_t, cvc_scalars, cls_ids):
        """Assemble x_i. Shapes: prev_conf/cur_conf [bs,T]; anchor_t [bs,T,box_dims];
        feat_t [bs,T,embed_dims]; cvc_scalars [bs,T,5]
        (c_geo, n_vis/ncam, n_hit/ncam, mean_matched_det_conf, max_matched_det_conf);
        cls_ids [bs,T] long. Returns [bs,T,in_dim]."""
        eps = 1e-6
        prev = prev_conf.clamp(eps, 1 - eps)
        state = torch.stack([
            prev_conf, cur_conf,
            torch.log(prev / (1 - prev)),          # logit(prev)
            cur_conf - prev_conf,                   # dconf
            anchor_t.new_zeros(prev_conf.shape),    # age  TODO(cvc): fill from bank age tracker
            anchor_t.new_zeros(prev_conf.shape),    # time_gap TODO(cvc): fill from metas.timestamp
        ], dim=-1)
        X, Y, Z = anchor_t[..., 0], anchor_t[..., 1], anchor_t[..., 2]
        logW, logL, logH = anchor_t[..., 3], anchor_t[..., 4], anchor_t[..., 5]
        # yaw: many TAO anchor layouts store (sin,cos) at 6,7; guard length
        if anchor_t.shape[-1] >= 8:
            syaw, cyaw = anchor_t[..., 6], anchor_t[..., 7]
        else:
            syaw = torch.zeros_like(X); cyaw = torch.ones_like(X)
        vel = torch.zeros_like(X)                   # TODO(cvc): ‖velocity‖ if present in anchor
        geom = torch.stack([X, Y, Z, logW, logL, logH, syaw, cyaw, vel], dim=-1)
        onehot = torch.zeros(*cls_ids.shape, NUM_CLASSES, device=cls_ids.device,
                             dtype=feat_t.dtype)
        onehot.scatter_(-1, cls_ids.clamp(0, NUM_CLASSES - 1).unsqueeze(-1), 1.0)
        cvc = torch.cat([cvc_scalars, onehot], dim=-1)
        idv = self.id_proj(feat_t)
        return torch.cat([state, geom, cvc, idv], dim=-1)


# ---------------------------------------------------------------------------
# Integration stubs — mirror the CCS hooks in instance_bank.py.
# TODO(cvc): move these onto InstanceBank once the design is approved.
# ---------------------------------------------------------------------------

def enable_cvc(bank, head, cams, dets, shift=(0.0, 0.0), ngc2off=None, conf=0.25):
    """Turn on the learned gate on an InstanceBank instance (analogue of enable_ccs).
    Registers projection geometry + independent 2D dets (RT-DETR), stores the trained head.
    Sets bank.cvc_enabled=True. Enforces mutual exclusion with CCS (ValueError so it survives
    `python -O`, which strips asserts)."""
    if getattr(bank, "ccs_enabled", False):
        raise ValueError("CVC and CCS are mutually exclusive (CCS already enabled on this bank)")
    bank.cvc_head = head
    bank.cvc_cams = {cid: dict(K=np.asarray(c["K"], np.float64),
                               R=np.asarray(c["R"], np.float64),
                               t=np.asarray(c["t"], np.float64),
                               W=int(c["W"]), H=int(c["H"])) for cid, c in cams.items()}
    bank.cvc_dets = dets
    bank.cvc_shift = (float(shift[0]), float(shift[1]))
    # Preserve the bank's correct default map when the caller omits ngc2off; NEVER wipe it to {}
    # (an empty map is identity -> no RT-DETR matches for the 5 remapped classes).
    if ngc2off is not None:
        bank.cvc_ngc2off = dict(ngc2off)
    bank.cvc_conf = float(conf)
    bank.cvc_enabled = True
    bank._cvc_frame = 0


@torch.no_grad()
def cvc_geom_scalars(cams, dets, frame, anchor_np, cls_np,
                     shift=(0.0, 0.0), ngc2off=None, conf=0.25):
    """Variant-A consensus scalars for one frame's temporal slots (standalone; the labeler
    calls this without an InstanceBank). Projection is byte-identical to
    InstanceBank._ccs_decay_vec (center Pc + footprint Pf, same-class point-in-box, visibility).

    cams     : {cam_id: dict(K(3x3), R(3x3), t(3,), W, H)}  world frame
    dets     : {"cam_id|frame": [[off_cls,x1,y1,x2,y2,conf], ...]}  official cls ids
    frame    : int video frame id (dets lookup key)
    anchor_np: [T, box_dims] RECENTERED-world anchors (X,Y,Z linear, W,L,H log) — shift added here
    cls_np   : [T] predicted class ids (NGC order)
    Returns  : [T, 5] float32 =
               [c_geo, n_vis/ncam, n_hit/ncam, mean_matched_det_conf, max_matched_det_conf]
    """
    ngc2off = ngc2off or {}
    a = np.asarray(anchor_np, np.float64)
    cls = np.asarray(cls_np)
    T = cls.shape[0]
    ncam = max(len(cams), 1)
    X = a[:, 0] + shift[0]
    Y = a[:, 1] + shift[1]
    Z = a[:, 2]
    Hh = np.exp(a[:, 5])
    off = np.array([ngc2off.get(int(c), int(c)) for c in cls])
    Pc = np.stack([X, Y, Z], 1)
    Pf = np.stack([X, Y, Z - Hh / 2.0], 1)
    vis = np.zeros(T, np.int32)
    hit = np.zeros(T, np.int32)
    sum_conf = np.zeros(T, np.float64)   # sum over hitting cams of best matched-det conf
    n_conf = np.zeros(T, np.int32)       # # cams contributing a matched det
    max_conf = np.zeros(T, np.float64)
    for cid, cam in cams.items():
        K = np.asarray(cam["K"], np.float64); R = np.asarray(cam["R"], np.float64)
        t = np.asarray(cam["t"], np.float64); W = int(cam["W"]); H = int(cam["H"])

        def proj(P):
            pc = P @ R.T + t
            d = pc[:, 2]
            uv = pc @ K.T
            ok = d > 1e-6
            u = np.where(ok, uv[:, 0] / np.where(ok, uv[:, 2], 1.0), -1e9)
            v = np.where(ok, uv[:, 1] / np.where(ok, uv[:, 2], 1.0), -1e9)
            return u, v, ok
        uc, vc, okc = proj(Pc)
        visible = okc & (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
        vis += visible.astype(np.int32)
        dl = dets.get(f"{cid}|{frame}")
        if not dl:
            continue
        arr = np.asarray(dl, np.float64)
        arr = arr[arr[:, 5] >= conf]
        if len(arr) == 0:
            continue
        uf, vf, okf = proj(Pf)
        dcls = arr[:, 0].astype(int)
        dx1, dy1, dx2, dy2, dcf = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        clsm = off[:, None] == dcls[None, :]
        cin = ((uc[:, None] >= dx1[None, :]) & (uc[:, None] <= dx2[None, :]) &
               (vc[:, None] >= dy1[None, :]) & (vc[:, None] <= dy2[None, :]))
        fin = (okf[:, None] &
               (uf[:, None] >= dx1[None, :]) & (uf[:, None] <= dx2[None, :]) &
               (vf[:, None] >= dy1[None, :]) & (vf[:, None] <= dy2[None, :]))
        match = clsm & (cin | fin)                            # (T, Ndet)
        anchor_hit = visible & match.any(axis=1)
        hit += anchor_hit.astype(np.int32)
        # best matched-det conf per anchor in THIS camera (0 where no match)
        cf = np.where(match, dcf[None, :], 0.0).max(axis=1)   # (T,)
        cf = np.where(anchor_hit, cf, 0.0)
        sum_conf += cf
        n_conf += (cf > 0).astype(np.int32)
        max_conf = np.maximum(max_conf, cf)
    seen = vis > 0
    cgeo = np.where(seen, hit / np.maximum(vis, 1), 0.0)
    mean_cf = np.where(n_conf > 0, sum_conf / np.maximum(n_conf, 1), 0.0)
    out = np.stack([cgeo, vis / ncam, hit / ncam, mean_cf, max_conf], axis=-1)
    return out.astype(np.float32)


# Runtime integration sketch for InstanceBank.cache() (see design §1) — NOT applied here.
# At inference-wire time, `enable_cvc` stores cams/dets/shift/ngc2off/conf on the bank and
# cache() calls the module-level cvc_geom_scalars(...) per current frame (bank._cvc_frame is
# set each step, exactly like set_ccs_frame). CVC and CCS are mutually exclusive.
#
#     if self.confidence is not None:
#         if getattr(self, "cvc_enabled", False):
#             cvc5 = torch.stack([                                   # [bs,T,5]
#                 torch.as_tensor(cvc_geom_scalars(
#                     self.cvc_cams, self.cvc_dets, self._cvc_frame,
#                     anchor[bi, :T].cpu().numpy(), cls_ids[bi, :T].cpu().numpy(),
#                     shift=self.cvc_shift, ngc2off=self.cvc_ngc2off, conf=self.cvc_conf))
#                 for bi in range(anchor.shape[0])]).to(confidence)
#             feat = self.cvc_head.build_feature(
#                 self.confidence, confidence[:, :T], anchor[:, :T],
#                 self.cached_feature, cvc5, cls_ids[:, :T])
#             decay = self.cvc_head(feat)                            # [bs,T] in (lo,hi)
#         elif self.ccs_enabled:
#             decay = self._ccs_decay_vec(anchor[:, :T], cls_ids[:, :T])
#         else:
#             decay = self.confidence_decay
#         confidence[:, :T] = torch.maximum(self.confidence * decay, confidence[:, :T])
