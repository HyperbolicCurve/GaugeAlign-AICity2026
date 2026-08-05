"""cvc_label_targets.py — build (temporal-instance -> GT-track survival) targets for CVC-Assoc.

Runs the FROZEN Sparse4D over the train scenes (RGB-only, recentered, temporal bank ON),
captures every temporal slot at each `instance_bank.cache()` call, matches each slot to the
scene GT by 3D-IoU, and emits a per-slot survival label y_i plus the feature vector x_i the
CVCAssocHead consumes. Output: one compact .npz per scene with (X, Y, meta) — the MLP then
trains offline in minutes on the dumped vectors (see docs/CVC_ASSOC_DESIGN.md §4-5).

Model build and dataset iteration mirror gaugealign/run_infer.py exactly, so the captured slot
statistics match what the gate sees at inference. Run in the inference env, e.g.:

  conda run -n gaugealign python gaugealign/assoc/cvc_label_targets.py \
    --config $NGC_MODEL_DIR/experiment.yaml --ckpt $NGC_MODEL_DIR/sparse4d_warehouse_v2.2_r50.pth \
    --anchor $NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy \
    --pkl work/w000_f0-900_c.pkl --scene_id 0 \
    --gt work/gt_train_track1.txt --out work/cvc_labels/w000.npz

This is exploratory code for the CVC-Assoc study (docs/EXPERIMENTS.md); it plays no part in the
submitted system.

Rule (design §4): y_i=1 iff slot matches a GT box (3D-IoU>=IOU_THR, same class) whose GT
track id PERSISTS >=K future frames; y_i=0 iff unmatched, or the matched GT track disappears
next frame (stale), or a higher-score slot already claimed that GT track this frame (dup).
"""
import os, sys, argparse, time, pickle
from collections import defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))               # <repo>/gaugealign/assoc
ROOT = os.path.dirname(os.path.dirname(HERE))                   # <repo>
REPO = os.environ.get("TAO_BACKEND", os.path.join(ROOT, "third_party", "tao_pytorch_backend"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))           # spatialai_data_utils stub
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)                                        # for cvc_assoc_head

from cvc_assoc_head import CVCAssocHead, cvc_geom_scalars, NUM_CLASSES
# NOTE: omegaconf + the nvidia_tao_pytorch model/dataloader imports are LAZY (inside build_model/
# main). They pull the full detector stack (heavy, env-specific); keeping them out of module scope
# lets the pure-numpy geometry helpers below be imported/tested standalone (e.g. cvc self-test).

# NGC model label order -> OFFICIAL Track1 class id (matches run_infer.py)
NGC2OFF = {0: 0, 1: 4, 2: 5, 3: 2, 4: 3, 5: 1, 6: 6}
IOU_THR = 0.3          # 3D-IoU (BEV*height) match threshold, per class
K_PERSIST = 5          # GT track must persist >=K future frames for y=1


# ----------------------------- geometry helpers -----------------------------
import math


def _yaw_of(box):
    """Yaw angle (rad) from a box. Anchor slots store (sin,cos) at [6],[7]; submission-format
    GT stores a single yaw angle at [6]. Fall back to 0 if neither is present."""
    if len(box) >= 8:
        return math.atan2(float(box[6]), float(box[7]))   # (sin, cos)
    if len(box) >= 7:
        return float(box[6])
    return 0.0


def _rect_corners(cx, cy, w, l, yaw):
    """4 BEV corners (CCW) of a box: local ±w/2 along heading-x, ±l/2 along heading-y, rotated."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = w / 2.0, l / 2.0
    out = []
    for lx, ly in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)):
        out.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return out


def _poly_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _convex_intersect_area(subject, clip):
    """Sutherland-Hodgman: area of the intersection of convex polygon `subject` clipped by the
    (CCW) convex polygon `clip`."""
    out = subject
    for i in range(len(clip)):
        A = clip[i]; B = clip[(i + 1) % len(clip)]
        ex, ey = B[0] - A[0], B[1] - A[1]           # clip edge
        inp = out; out = []
        if not inp:
            return 0.0
        for j in range(len(inp)):
            P = inp[j]; Q = inp[(j + 1) % len(inp)]
            dP = ex * (P[1] - A[1]) - ey * (P[0] - A[0])   # >=0 => inside (CCW)
            dQ = ex * (Q[1] - A[1]) - ey * (Q[0] - A[0])
            if dP >= 0:
                out.append(P)
                if dQ < 0:
                    tt = dP / (dP - dQ)
                    out.append((P[0] + tt * (Q[0] - P[0]), P[1] + tt * (Q[1] - P[1])))
            elif dQ >= 0:
                tt = dP / (dP - dQ)
                out.append((P[0] + tt * (Q[0] - P[0]), P[1] + tt * (Q[1] - P[1])))
    return _poly_area(out)


def _bev_iou(a, b):
    """Oriented 3D IoU (rotated BEV footprint x height overlap) between two [x,y,z,w,l,h,...]
    boxes. `a` is an anchor slot (LINEAR w,l,h at 3,4,5 — caller decodes log->linear; sin,cos at
    6,7); `b` is a submission-format GT box (linear w,l,h, yaw angle at 6). Proper oriented IoU
    (not the axis-aligned proxy, which over-estimates overlap for rotated/elongated objects and
    manufactures false matches). Returns 0 on any non-finite/non-positive size."""
    aw, al, ah = float(a[3]), float(a[4]), float(a[5])
    bw, bl, bh = float(b[3]), float(b[4]), float(b[5])
    for d in (aw, al, ah, bw, bl, bh):
        if not math.isfinite(d) or d <= 0.0:
            return 0.0
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    # cheap center-distance reject before the polygon clip
    ra = 0.5 * math.hypot(aw, al); rb = 0.5 * math.hypot(bw, bl)
    if (ax - bx) ** 2 + (ay - by) ** 2 > (ra + rb) ** 2:
        return 0.0
    ca = _rect_corners(ax, ay, aw, al, _yaw_of(a))
    cb = _rect_corners(bx, by, bw, bl, _yaw_of(b))
    inter_bev = _convex_intersect_area(ca, cb)
    if inter_bev <= 0.0:
        return 0.0
    iz = max(0.0, min(az + ah / 2, bz + bh / 2) - max(az - ah / 2, bz - bh / 2))
    inter = inter_bev * iz
    va, vb = aw * al * ah, bw * bl * bh
    return inter / max(va + vb - inter, 1e-6)


def load_gt(gt_path, scene_id):
    """Parse submission-format GT -> {frame: [(track_id, off_cls, box7)]} and the
    per-track set of frames it appears in (for the persistence test)."""
    per_frame = defaultdict(list)
    track_frames = defaultdict(set)
    with open(gt_path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 11 or int(p[0]) != scene_id:
                continue
            off_cls, tid, fr = int(p[1]), int(p[2]), int(p[3])
            box = np.array([float(p[4]), float(p[5]), float(p[6]),
                            float(p[7]), float(p[8]), float(p[9]), float(p[10])], np.float32)
            per_frame[fr].append((tid, off_cls, box))
            track_frames[tid].add(fr)
    return per_frame, track_frames


# --------------------------- cache() capture hook ---------------------------
class CacheProbe:
    """Wraps InstanceBank.cache to snapshot, per call, the pre-decay previous score, the
    current score, the kept temporal anchors/features, and the bank-assigned instance ids.
    We patch cache() to record its inputs/outputs rather than reimplement it."""
    def __init__(self, bank):
        self.bank = bank
        self.records = []          # one dict per frame
        self._orig = bank.cache
        bank.cache = self._wrapped

    def _wrapped(self, instance_feature, anchor, confidence, *a, **kw):
        # snapshot the PREVIOUS-frame cached score aligned to this frame's slots
        T = self.bank.num_temp_instances
        prev_conf = None if self.bank.confidence is None else self.bank.confidence.detach().clone()
        cls_ids = confidence.argmax(dim=-1).detach().clone()          # [bs, num_anchor]
        cur_conf_full = confidence.max(dim=-1).values.sigmoid().detach().clone()
        # Snapshot the CURRENT-frame refined inputs for the carried temporal slots BEFORE topk,
        # so feat_t/anchor_t/cur_conf/cls_ids are all [:, :T] of the same current-frame tensors —
        # exactly what InstanceBank._cvc_decay_vec sees at inference (decay is computed pre-topk).
        # (Previously feat_t used the POST-topk cached_feature, which is misaligned with
        # anchor[:, :T]; that would train the head on features it never sees at inference.)
        feat_t = instance_feature[:, :T].detach().float().cpu().numpy()  # [bs,T,embed]
        anchor_t = anchor[:, :T].detach().float().cpu().numpy()          # [bs,T,box]
        ret = self._orig(instance_feature, anchor, confidence, *a, **kw)
        rec = dict(
            anchor_t=anchor_t,
            feat_t=feat_t,
            cur_conf=cur_conf_full[:, :T].cpu().numpy(),                 # [bs,T]
            prev_conf=(prev_conf.cpu().numpy() if prev_conf is not None else None),
            cls_ids=cls_ids[:, :T].cpu().numpy(),                        # [bs,T]
            inst_id=(self.bank.instance_id[:, :T].detach().cpu().numpy()
                     if getattr(self.bank, "instance_id", None) is not None else None),
        )
        self.records.append(rec)
        return ret

    def restore(self):
        self.bank.cache = self._orig


# ------------------------------ collate (from run_infer) --------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    data = {k: [item[k] for item in batch] for k in batch[0].keys()}
    for k, op in (("img", torch.stack), ("projection_mat", None),
                  ("image_wh", None), ("timestamp", torch.stack)):
        if k not in data:
            continue
        if k == "img" and isinstance(data["img"][0], torch.Tensor):
            data["img"] = torch.stack(data["img"], dim=0)
        elif k in ("projection_mat", "image_wh"):
            data[k] = torch.stack([torch.as_tensor(x) for x in data[k]], dim=0)
        elif k == "timestamp":
            data[k] = torch.stack([torch.as_tensor(x) for x in data[k]], dim=0)
    if "focal" in data:
        data["focal"] = torch.cat([torch.as_tensor(x.flatten()) for x in data["focal"]], dim=0)
    return data


def to_cuda(batch):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cuda()
        elif isinstance(v, list) and len(v) and isinstance(v[0], torch.Tensor):
            out[k] = [x.cuda() for x in v]
        else:
            out[k] = v
    return out


def _file_sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _head_id_hash(head):
    """Hash the FROZEN id_proj weights that get baked into every X vector. The trainer refuses
    npz whose id_proj hash disagrees (fix F): mixing features from two different random id_proj
    inits silently corrupts the training set."""
    import hashlib
    h = hashlib.sha1()
    sd = head.state_dict()
    for k in ("id_proj.weight", "id_proj.bias"):
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def build_model(cfg, ckpt_path, anchor):
    """Load the FROZEN base model. STRICT: any substantive missing/unexpected key (i.e. not a
    BN num_batches_tracked buffer) is a fatal load mismatch — a silently-partial detector would
    label the whole training set with garbage slot statistics. Also returns the ckpt hash so the
    npz records exactly which weights produced its features (provenance, fix F)."""
    from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import Sparse4D  # lazy (heavy)
    cfg.model.head.instance_bank.anchor = anchor
    model = Sparse4D(config=cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    subst_miss = [m for m in miss if "num_batches_tracked" not in m]
    subst_unexp = [u for u in unexp if "num_batches_tracked" not in u]
    print(f"load_state_dict: missing={len(subst_miss)} unexpected={len(subst_unexp)} "
          f"(ignored num_batches_tracked)", flush=True)
    if subst_miss or subst_unexp:
        raise RuntimeError(
            "ckpt load mismatch — refusing to label with a partially-loaded detector.\n"
            f"  missing[:20]={subst_miss[:20]}\n  unexpected[:20]={subst_unexp[:20]}")
    ckpt_hash = _file_sha1(ckpt_path)
    print(f"ckpt sha1[:16]={ckpt_hash}", flush=True)
    return model.eval().cuda(), ckpt_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="BASE pretrained ckpt (decision 4: label FT-independent, NOT the FT ckpt)")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--gt", required=True, help="submission-format GT (gt_train_track1.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--half_res", action="store_true")
    # Variant-A consensus features (decision 1). Both required for real cvc scalars;
    # if omitted the labeler still runs but the 5 cvc dims are zeros (dry-run only).
    ap.add_argument("--dets", type=str, default="",
                    help="cached RT-DETR dets json {'cam|frame':[[off_cls,x1,y1,x2,y2,conf],..]}")
    ap.add_argument("--calib", type=str, default="", help="calibration.json for reprojection")
    ap.add_argument("--det_conf", type=float, default=0.25, help="min RT-DETR 2D det conf")
    ap.add_argument("--head_init", type=str, default="",
                    help="cvc_head_init.pt (make-init). REQUIRED for real runs (omit only with "
                         "--dry_run): build_feature bakes id_proj(feat_t) into X, so the SAME "
                         "frozen id_proj must be reused across all scenes + inference. Fresh "
                         "random head -> inconsistent X.")
    ap.add_argument("--dry_run", action="store_true",
                    help="debug only: allow missing --head_init/--dets/--calib (X is NOT "
                         "train-consistent and cvc scalars are zeros; never use for real labels).")
    args = ap.parse_args()

    # ---- provenance gate (fix F): real runs must have a shared frozen head AND consensus dets ----
    if not args.dry_run:
        if not args.head_init:
            raise SystemExit("ERROR: --head_init is required for real labeling runs (pass "
                             "--dry_run to override for debugging only).")
        if not (args.calib and args.dets):
            raise SystemExit("ERROR: --calib and --dets are required for real labeling runs "
                             "(Variant-A consensus features). Pass --dry_run to override.")

    # lazy heavy imports (keep module scope light so the geometry helpers import standalone)
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import Omniverse3DDetTrackDataset
    from nvidia_tao_pytorch.cv.sparse4d.dataloader.transforms import (
        LoadMultiViewImageFromFiles, AICitySparse4DAdaptor, Compose)
    from nvidia_tao_pytorch.cv.sparse4d.dataloader.augment import (
        ResizeCropFlipImage, NormalizeMultiviewImage)

    cfg = OmegaConf.load(args.config)
    if args.half_res:
        cfg.dataset.augmentation.image_size = [540, 960]
    classes = list(cfg.dataset.classes)
    aug = OmegaConf.to_container(cfg.dataset.augmentation, resolve=True)
    norm = OmegaConf.to_container(cfg.dataset.normalize, resolve=True)
    transforms = Compose([
        LoadMultiViewImageFromFiles(to_float32=True, h5_file=False),
        ResizeCropFlipImage(),
        NormalizeMultiviewImage(mean=norm["mean"], std=norm["std"], to_rgb=norm["to_rgb"]),
        AICitySparse4DAdaptor(),
    ])
    ds = Omniverse3DDetTrackDataset(
        data_root="", anno_file=args.pkl, classes=classes, test_mode=True,
        augmentation=aug, tracking=True, tracking_threshold=0.2,
        transforms=transforms, with_velocity=True)
    actual_frames = [info.get("actual_frame", info["frame_idx"]) for info in ds.data_infos]
    with open(args.pkl, "rb") as f:
        shift = pickle.load(f).get("metadata", {}).get("recenter_shift", [0.0, 0.0])
    print(f"scene {args.scene_id}: {len(ds)} frames, recenter_shift={shift}", flush=True)

    per_frame_gt, track_frames = load_gt(args.gt, args.scene_id)
    print(f"GT: {len(track_frames)} tracks over {len(per_frame_gt)} frames", flush=True)

    # ---- Variant-A consensus source: calibration cameras + cached RT-DETR dets ----
    cams, dets = None, {}
    if args.calib and args.dets:
        import json
        cal = json.load(open(args.calib))
        cams = {}
        for s in cal["sensors"]:
            if s.get("type") != "camera":
                continue
            ext = np.array(s["extrinsicMatrix"], np.float64)
            W = H = None
            for at in s.get("attributes", []):
                if at["name"] == "frameWidth":
                    W = int(float(at["value"]))
                if at["name"] == "frameHeight":
                    H = int(float(at["value"]))
            cams[s["id"]] = dict(K=np.array(s["intrinsicMatrix"], np.float64),
                                 R=ext[:3, :3], t=ext[:3, 3], W=W or 1920, H=H or 1080)
        dets = json.load(open(args.dets))
        print(f"Variant-A: {len(cams)} cams, {len(dets)} det-tiles, conf>={args.det_conf}", flush=True)
    else:
        print("WARNING: no --calib/--dets -> cvc scalars = ZEROS (dry-run; not for training)", flush=True)

    model, ckpt_hash = build_model(cfg, args.ckpt, args.anchor)
    bank = model.head.instance_bank
    probe = CacheProbe(bank)
    # Feature builder: MUST be the shared frozen init head (see --head_init help). Falls back to
    # a fresh random head only for --dry_run debugging (X then not train-consistent).
    if args.head_init:
        from cvc_train_head import load_head
        head, _ = load_head(args.head_init)
        print(f"loaded shared init head <- {args.head_init} (in_dim={head.in_dim})", flush=True)
    else:
        print("WARNING(dry_run): no --head_init -> fresh random id_proj; X NOT consistent",
              flush=True)
        head = CVCAssocHead(embed_dims=bank.embed_dims if hasattr(bank, "embed_dims") else 256)
    # FROZEN: id_proj/MLP never train here. Also required for the pass-2 build_feature call — a
    # grad-enabled id_proj would make build_feature's output require grad and .numpy() would raise
    # after the full labeling pass (codex CRITICAL). eval() + no-grad params => plain numpy output.
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    head_hash = _head_id_hash(head)
    print(f"id_proj sha1[:16]={head_hash}  has_consensus={cams is not None}", flush=True)

    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(ds), batch_size=1, drop_last=False),
        num_workers=args.num_workers, collate_fn=collate_fn)

    # ---- pass 1: frozen forward, temporal bank on; probe records every cache() ----
    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if not batch:
                probe.records.append(None); continue
            model.simple_test(to_cuda(batch)["img"], to_cuda(batch))
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(ds)}] {(idx+1)/(time.time()-t0):.2f} fps", flush=True)
    probe.restore()
    assert len(probe.records) == len(ds), (len(probe.records), len(ds))

    # ---- pass 2: match slots to GT, assign survival label, assemble features ----
    X, Y, META = [], [], []
    for idx, rec in enumerate(probe.records):
        if rec is None:
            continue
        # First frame: self.confidence was None so cache() never runs the decay gate; its slots
        # would train the head on inputs it never sees at inference -> skip (fix D).
        if rec["prev_conf"] is None:
            continue
        fr = int(actual_frames[idx])
        gts = per_frame_gt.get(fr, [])
        # the next K SAMPLED scene frames (inference cadence), for the causal persistence test
        next_frames = [int(f) for f in actual_frames[idx + 1: idx + 1 + K_PERSIST]]
        bs, T = rec["cur_conf"].shape
        # shift bank anchors back into GT world frame before matching
        anc = rec["anchor_t"].copy()
        anc[..., 0] += shift[0]; anc[..., 1] += shift[1]
        # CRITICAL: anchor W,L,H are stored as LOG dims (see instance_bank._ccs_decay_vec:289
        # `np.exp(a[...,5])` and cvc_geom_scalars:154). GT boxes (gt_train_track1.txt) are LINEAR
        # meters. Decode log->linear here so _bev_iou compares like-for-like units; without this
        # the 3D-IoU match is garbage and the survival labels Y are noise. NOTE: build_feature and
        # cvc_geom_scalars below read the UNTOUCHED rec["anchor_t"] (which correctly stays log), so
        # this decode is isolated to the matching path only.
        anc[..., 3:6] = np.exp(anc[..., 3:6])
        prev = rec["prev_conf"]
        for bi in range(bs):
            claimed = {}                       # gt_track_id -> best (score, slot) this frame
            slot_match = [None] * T            # slot -> (gt_tid, iou)
            order = np.argsort(-rec["cur_conf"][bi])   # high score first, for dup resolution
            for si in order:
                off_cls = NGC2OFF.get(int(rec["cls_ids"][bi, si]), int(rec["cls_ids"][bi, si]))
                best_iou, best_tid = 0.0, None
                for tid, gcls, gbox in gts:
                    if gcls != off_cls:
                        continue
                    iou = _bev_iou(anc[bi, si], gbox)
                    if iou > best_iou:
                        best_iou, best_tid = iou, tid
                if best_tid is not None and best_iou >= IOU_THR:
                    slot_match[si] = (best_tid, best_iou)
                    if best_tid not in claimed:
                        claimed[best_tid] = si          # first (highest score) wins the track
            # survival label per slot
            y_arr = np.zeros(T, np.int8)
            for si in range(T):
                m = slot_match[si]
                if m is None:
                    continue                            # unmatched -> decay (y=0)
                tid, _ = m
                if claimed.get(tid) != si:
                    continue                            # duplicate of a higher-score slot (y=0)
                # PERSISTENCE (fix C): the decay gate is applied per-step, so the label must reflect
                # near-term survival on the SAMPLED cadence, not "appears anywhere later". Require
                # the GT track to be present in the immediate next sampled frame AND in every one of
                # the next K sampled frames (all remaining if fewer than K before scene end). The old
                # `count any future frame >= K` rewarded tracks that vanish next step but reappear
                # after a long gap -> exactly the ghost the gate should suppress.
                tf = track_frames[tid]
                if (next_frames and next_frames[0] in tf and
                        sum(1 for nf in next_frames if nf in tf) >= min(K_PERSIST, len(next_frames))):
                    y_arr[si] = 1                       # persistent
                # else y stays 0 (stale / vanishes next step / gap)
            # Variant-A consensus scalars: pass RECENTERED anchors (cvc_geom_scalars adds
            # `shift` internally, same as _ccs_decay_vec). [T,5], zeros if no dets provided.
            if cams is not None:
                cvc5 = cvc_geom_scalars(cams, dets, fr, rec["anchor_t"][bi], rec["cls_ids"][bi],
                                        shift=shift, ngc2off=NGC2OFF, conf=args.det_conf)
            else:
                cvc5 = np.zeros((T, 5), np.float32)
            # build_feature gets RECENTERED anchors (matches what cache() sees at inference)
            pconf = prev[bi]                            # [T] (prev is never None: first frame skipped)
            feats = head.build_feature(
                torch.as_tensor(pconf).view(1, T).float(),
                torch.as_tensor(rec["cur_conf"][bi]).view(1, T).float(),
                torch.as_tensor(rec["anchor_t"][bi]).view(1, T, -1).float(),
                torch.as_tensor(rec["feat_t"][bi]).view(1, T, -1).float(),
                torch.as_tensor(cvc5).view(1, T, -1).float(),
                torch.as_tensor(rec["cls_ids"][bi]).view(1, T).long(),
            ).view(T, -1).detach().cpu().numpy()        # head params frozen -> no-grad numpy
            for si in range(T):
                X.append(feats[si].astype(np.float32)); Y.append(int(y_arr[si]))
                META.append((args.scene_id, fr, si, int(rec["cls_ids"][bi, si])))

    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int8)
    META = np.asarray(META, np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Provenance (fix F): the trainer joins scenes only when id_proj hash agrees and consensus is
    # present, and records which base ckpt produced the features.
    np.savez_compressed(args.out, X=X, Y=Y, meta=META, in_dim=head.in_dim,
                        head_hash=np.str_(head_hash), ckpt_hash=np.str_(ckpt_hash),
                        has_consensus=np.int8(1 if cams is not None else 0))
    pos = int(Y.sum())
    print(f"wrote {len(Y)} slot-samples ({pos} pos / {len(Y)-pos} neg, "
          f"{100*pos/max(len(Y),1):.1f}% pos) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
