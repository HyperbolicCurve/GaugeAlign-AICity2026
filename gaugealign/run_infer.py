"""Zero-shot Sparse4D inference on an OVPKL scene -> 11-col track1.txt.

Runs the TAO Sparse4D model (pretrained .pth) in plain PyTorch over the
Omniverse3DDetTrackDataset (test_mode, tracking) and writes world-frame
3D boxes + temporal track ids in the official track1 format:
  scene_id class_id object_id frame_id x y z width length height yaw
"""
import os, sys, argparse, time
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/gaugealign
ROOT = os.path.dirname(HERE)                               # <repo>
# Patched NVIDIA TAO PyTorch backend (see docs/INSTALL.md). Override the location
# with TAO_BACKEND=/path/to/tao_pytorch_backend if it lives outside the repo.
REPO = os.environ.get("TAO_BACKEND", os.path.join(ROOT, "third_party", "tao_pytorch_backend"))
if not os.path.isdir(REPO):
    raise SystemExit(
        f"TAO PyTorch backend not found at {REPO}\n"
        "Run  bash scripts/setup_tao_backend.sh  (or set TAO_BACKEND=...); see docs/INSTALL.md")
sys.path.insert(0, os.path.join(ROOT, "third_party"))  # spatialai_data_utils stub
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "assoc"))        # cvc_assoc_head / cvc_train_head (CVC path)

from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import Sparse4D
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import Omniverse3DDetTrackDataset
from nvidia_tao_pytorch.cv.sparse4d.dataloader.transforms import (
    LoadMultiViewImageFromFiles, AICitySparse4DAdaptor, Compose)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.augment import (
    ResizeCropFlipImage, NormalizeMultiviewImage)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    data = {k: [item[k] for item in batch] for k in batch[0].keys()}
    if "img" in data and isinstance(data["img"][0], torch.Tensor):
        data["img"] = torch.stack(data["img"], dim=0)
    if "projection_mat" in data:
        data["projection_mat"] = torch.stack([torch.as_tensor(x) for x in data["projection_mat"]], dim=0)
    if "image_wh" in data:
        data["image_wh"] = torch.stack([torch.as_tensor(x) for x in data["image_wh"]], dim=0)
    if "focal" in data:
        data["focal"] = torch.cat([torch.as_tensor(x.flatten()) for x in data["focal"]], dim=0)
    if "timestamp" in data:
        data["timestamp"] = torch.stack([torch.as_tensor(x) for x in data["timestamp"]], dim=0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--swap_wl", action="store_true", help="swap width/length columns")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--to_rgb", type=int, default=None, help="override normalize.to_rgb (0/1)")
    ap.add_argument("--topk", type=int, default=0, help="keep top-K dets per frame (0=off)")
    ap.add_argument("--dec_thr", type=float, default=None, help="override decoder.score_threshold")
    ap.add_argument("--remap", action="store_true", help="remap NGC labels -> official Track1 class ids")
    ap.add_argument("--half_res", action="store_true",
                    help="frames pre-decoded at 960x540 (intrinsic already halved in pkl); no resize")
    ap.add_argument("--adabn", type=int, default=0,
                    help="AdaBN: adapt backbone BN stats over first N frames of the scene (0=off)")
    # ---- AssA (tracking) knobs on the temporal instance bank ----
    ap.add_argument("--conf_decay", type=float, default=None,
                    help="instance_bank.confidence_decay (0.8 default; higher=IDs survive occlusion longer)")
    ap.add_argument("--num_temp", type=int, default=None,
                    help="instance_bank.num_temp_instances (600 default; up to num_anchor 900)")
    ap.add_argument("--max_time_interval", type=float, default=None,
                    help="instance_bank max_time_interval (2 default; higher=survive longer gaps)")
    ap.add_argument("--cam_keep", type=str, default="",
                    help="CCS camera mask: comma cam indices to KEEP (''=all). Subsets img/proj/image_wh dim=1.")
    # ---- CCS (STEP 2): geometry-conditioned per-anchor decay on the instance bank ----
    ap.add_argument("--ccs_dets", type=str, default="",
                    help="path to cached RT-DETR dets json {cam_id|frame:[[off_cls,x1,y1,x2,y2,conf],...]}; enables CCS")
    ap.add_argument("--ccs_calib", type=str, default="",
                    help="path to calibration.json for CCS reprojection (required with --ccs_dets)")
    ap.add_argument("--ccs_decay_lo", type=float, default=0.85,
                    help="decay for visible-but-unconfirmed tracks (c_geo=0)")
    ap.add_argument("--ccs_decay_hi", type=float, default=0.93,
                    help="decay for fully-confirmed tracks (c_geo=1)")
    ap.add_argument("--ccs_conf", type=float, default=0.25,
                    help="min RT-DETR 2D det confidence to count as a confirmation")
    ap.add_argument("--ccs_min_cams", type=int, default=3,
                    help="STEP 2b VISIBILITY gate M: box must be visible in >=M cameras to c_geo-modulate")
    ap.add_argument("--ccs_cover", type=str, default="",
                    help="STEP 2b CLASS-COVERAGE gate: comma-sep OFFICIAL class ids eligible for "
                         "c_geo modulation in this scene (RT-DETR coverage>=tau); empty = all classes eligible")
    ap.add_argument("--ccs_min_scene_cams", type=int, default=0,
                    help="STEP 2c SCENE gate K: if scene has <K cameras, CCS is a complete no-op "
                         "(scalar decay, byte-identical to base). 0 disables the scene gate.")
    # ---- CVC-Assoc (OUR architecture): LEARNED per-anchor survival gate on the bank ----
    # Off by default (--cvc_head ""): byte-identical to base/CCS. Mutually exclusive with CCS.
    ap.add_argument("--cvc_head", type=str, default="",
                    help="trained CVCAssocHead (cvc_head_trained.pt); enables the learned gate")
    ap.add_argument("--cvc_dets", type=str, default="",
                    help="cached RT-DETR dets json for CVC consensus features (required with --cvc_head)")
    ap.add_argument("--cvc_calib", type=str, default="",
                    help="calibration.json for CVC reprojection (required with --cvc_head)")
    ap.add_argument("--cvc_conf", type=float, default=0.25, help="min RT-DETR 2D det conf for CVC")
    args = ap.parse_args()
    keep_cams = sorted(int(x) for x in args.cam_keep.split(",") if x.strip() != "") if args.cam_keep.strip() else None

    cfg = OmegaConf.load(args.config)
    cfg.model.head.instance_bank.anchor = args.anchor
    if args.conf_decay is not None:
        cfg.model.head.instance_bank.confidence_decay = args.conf_decay
    if args.num_temp is not None:
        cfg.model.head.instance_bank.num_temp_instances = args.num_temp
    if args.max_time_interval is not None:
        OmegaConf.update(cfg, "model.head.instance_bank.max_time_interval", args.max_time_interval, force_add=True)
    print(f"[track knobs] conf_decay={cfg.model.head.instance_bank.confidence_decay} "
          f"num_temp={cfg.model.head.instance_bank.num_temp_instances}", flush=True)
    if args.half_res:
        cfg.dataset.augmentation.image_size = [540, 960]  # -> get_augmentation resize=1.0
    if args.dec_thr is not None:
        cfg.model.head.decoder.score_threshold = args.dec_thr
    print("decoder.score_threshold =", cfg.model.head.decoder.score_threshold, flush=True)

    classes = list(cfg.dataset.classes)
    print("classes:", classes, flush=True)
    aug = OmegaConf.to_container(cfg.dataset.augmentation, resolve=True)
    norm = OmegaConf.to_container(cfg.dataset.normalize, resolve=True)
    if args.to_rgb is not None:
        norm["to_rgb"] = bool(args.to_rgb)
    print("normalize.to_rgb =", norm["to_rgb"], flush=True)

    transforms = Compose([
        LoadMultiViewImageFromFiles(to_float32=True, h5_file=False),
        ResizeCropFlipImage(),
        NormalizeMultiviewImage(mean=norm["mean"], std=norm["std"], to_rgb=norm["to_rgb"]),
        AICitySparse4DAdaptor(),
    ])

    ds = Omniverse3DDetTrackDataset(
        data_root="", anno_file=args.pkl, classes=classes,
        test_mode=True, augmentation=aug, tracking=True, tracking_threshold=0.2,
        transforms=transforms, with_velocity=True,
    )
    print("dataset frames:", len(ds), flush=True)

    # keep a map: dataset order index -> actual video frame number
    actual_frames = [info.get("actual_frame", info["frame_idx"]) for info in ds.data_infos]

    # world re-center shift stored in pkl metadata -> add back to predictions
    import pickle as _pk
    with open(args.pkl, "rb") as _f:
        _meta = _pk.load(_f).get("metadata", {})
    shift = _meta.get("recenter_shift", [0.0, 0.0])
    print("recenter_shift (added back to preds):", shift, flush=True)

    # NGC model label order -> OFFICIAL Track1 class id
    # NGC: [person,gr1_t2,agility_digit,nova_carter,transporter,forklift,pallet_truck]
    # off: [person,forklift,nova_carter,transporter,gr1_t2,agility_digit,pallet_truck]
    NGC2OFF = {0: 0, 1: 4, 2: 5, 3: 2, 4: 3, 5: 1, 6: 6}

    sampler = torch.utils.data.SequentialSampler(ds)
    bsam = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=bsam, num_workers=args.num_workers,
                                         collate_fn=collate_fn)

    model = Sparse4D(config=cfg)
    # The raw checkpoint keys (img_backbone.*, img_neck.*, head.*, depth_branch.*)
    # match the raw Sparse4D module. load_pretrained_weights() adds a "model." prefix
    # for the Lightning wrapper (Sparse4DPlModel.model) which we do NOT use here, so
    # load directly and strip any "model." prefix.
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "num_batches_tracked" not in m]
    print(f"load_state_dict: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if miss[:10]:
        print("  sample missing:", miss[:10], flush=True)
    if unexp[:10]:
        print("  sample unexpected:", unexp[:10], flush=True)
    model.eval().cuda()

    # ---- AdaBN: adapt img_backbone BatchNorm running stats to THIS scene (causal,
    # label-free) over the first N frames, then freeze the adapted stats. Attacks the
    # sim->real BN covariate shift on the real test RGB. ----
    if args.adabn > 0:
        # WARNING (online-bonus safety): AdaBN accumulates BN stats over the first N frames of the
        # scene BEFORE emitting boxes for frame 0 — i.e. it reads ahead. That is NOT strictly
        # causal, so it FORFEITS the online/causal +10% HOTA bonus. Never combine with a submission
        # meant to score as online (CVC is our online contribution). Use only for offline ablations.
        print(f"[AdaBN][WARN] adabn={args.adabn}: adapts BN over warmup frames (reads ahead) -> "
              f"NOT causal, forfeits the online +10% bonus. Do NOT use for the online submission.",
              flush=True)
        # backbone uses FrozenBatchNorm2d (fixed running_mean/var, no train-mode update),
        # so recompute those buffers from THIS scene's real activations via forward hooks.
        bns = [m for m in model.img_backbone.modules()
               if hasattr(m, "running_mean") and hasattr(m, "running_var")]
        acc = {id(m): [0.0, 0.0, 0] for m in bns}  # sum, sqsum, count(per-channel)

        def mk_hook(m):
            def hook(mod, inp, out):
                x = inp[0].detach().float()
                a = acc[id(m)]
                a[0] = a[0] + x.sum(dim=(0, 2, 3)).double()
                a[1] = a[1] + (x * x).sum(dim=(0, 2, 3)).double()
                a[2] = a[2] + x.shape[0] * x.shape[2] * x.shape[3]
            return hook
        hooks = [m.register_forward_hook(mk_hook(m)) for m in bns]
        seen = 0
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if not batch or seen >= args.adabn:
                    break
                b = to_cuda(batch)
                model.extract_feat(b["img"], False, b)  # backbone+neck -> triggers hooks
                seen += 1
        for h in hooks:
            h.remove()
        for m in bns:
            s, sq, c = acc[id(m)]
            if c > 0:
                mean = (s / c).float()
                var = (sq / c).float() - mean ** 2
                m.running_mean.copy_(mean.to(m.running_mean.device))
                m.running_var.copy_(var.clamp_min(1e-5).to(m.running_var.device))
        model.head.instance_bank.reset()  # clear temporal state from warmup
        print(f"[AdaBN] recomputed {len(bns)} FrozenBN layers over {seen} warmup frames", flush=True)

    # ---- CCS (STEP 2): enable geometry-conditioned per-anchor decay ----
    ccs_on = bool(args.ccs_dets)
    if ccs_on:
        import json as _json
        assert args.ccs_calib, "--ccs_dets requires --ccs_calib"
        _cal = _json.load(open(args.ccs_calib))
        _cams = {}
        for _s in _cal["sensors"]:
            if _s.get("type") != "camera":
                continue
            _ext = np.array(_s["extrinsicMatrix"], np.float64)
            _W = _H = None
            for _at in _s.get("attributes", []):
                if _at["name"] == "frameWidth":
                    _W = int(float(_at["value"]))
                if _at["name"] == "frameHeight":
                    _H = int(float(_at["value"]))
            _cams[_s["id"]] = dict(K=np.array(_s["intrinsicMatrix"], np.float64),
                                   R=_ext[:3, :3], t=_ext[:3, 3], W=_W or 1920, H=_H or 1080)
        _dets = _json.load(open(args.ccs_dets))
        _cover = sorted(int(x) for x in args.ccs_cover.split(",") if x.strip() != "") if args.ccs_cover.strip() else None
        model.head.instance_bank.enable_ccs(
            _cams, _dets, shift=shift, decay_lo=args.ccs_decay_lo,
            decay_hi=args.ccs_decay_hi, conf=args.ccs_conf, ngc2off=NGC2OFF,
            min_cams=args.ccs_min_cams, cover_classes=_cover,
            min_scene_cams=args.ccs_min_scene_cams)
        _ib = model.head.instance_bank
        if _ib.ccs_scene_disabled:
            print(f"[CCS] SCENE-GATE: n_cams={_ib.ccs_n_cams} < K={args.ccs_min_scene_cams} "
                  f"-> CCS DISABLED (scalar cd, byte-identical to base)", flush=True)
        else:
            print(f"[CCS] enabled: {len(_cams)} cams (>=K={args.ccs_min_scene_cams}), {len(_dets)} det-tiles, "
                  f"decay∈[{args.ccs_decay_lo},{args.ccs_decay_hi}] conf>={args.ccs_conf} "
                  f"min_cams={args.ccs_min_cams} cover={_cover if _cover is not None else 'ALL'}", flush=True)

    # ---- CVC-Assoc (OUR architecture): enable the LEARNED per-anchor survival gate ----
    cvc_on = bool(args.cvc_head)
    if cvc_on:
        assert not ccs_on, "--cvc_head and --ccs_dets are mutually exclusive"
        assert args.cvc_dets and args.cvc_calib, "--cvc_head requires --cvc_dets and --cvc_calib"
        # CVC is the ONLINE contribution; AdaBN reads ahead and forfeits the online bonus (fix M).
        assert args.adabn == 0, ("--cvc_head is the online/causal contribution and MUST NOT be "
                                 "combined with --adabn (which reads warmup frames ahead and "
                                 "forfeits the +10% online bonus).")
        if not args.remap:
            # CVC consensus uses NGC2OFF internally, but submission rows are written with RAW NGC
            # ids unless --remap is set. The val-gate/official submission needs official ids.
            print("[CVC][WARN] --remap NOT set: output class ids will be RAW NGC, not official "
                  "Track1 ids. The CVC val-gate and any submission almost certainly need --remap.",
                  flush=True)
        import json as _json
        from cvc_assoc_head import enable_cvc as _enable_cvc
        from cvc_train_head import load_head as _load_head
        _cal = _json.load(open(args.cvc_calib))
        _vcams = {}
        for _s in _cal["sensors"]:
            if _s.get("type") != "camera":
                continue
            _ext = np.array(_s["extrinsicMatrix"], np.float64)
            _W = _H = None
            for _at in _s.get("attributes", []):
                if _at["name"] == "frameWidth":
                    _W = int(float(_at["value"]))
                if _at["name"] == "frameHeight":
                    _H = int(float(_at["value"]))
            _vcams[_s["id"]] = dict(K=np.array(_s["intrinsicMatrix"], np.float64),
                                    R=_ext[:3, :3], t=_ext[:3, 3], W=_W or 1920, H=_H or 1080)
        _vdets = _json.load(open(args.cvc_dets))
        _vhead, _vcfg = _load_head(args.cvc_head)
        # ---- validate the head BEFORE it touches the GPU / the bank (fix K) ----
        # A corrupt or mis-ranged head could silently push the survival gate outside the proven
        # floor band; refuse rather than risk regressing below the scalar floor at submission time.
        _lo, _hi, _bd = float(_vhead.decay_lo), float(_vhead.decay_hi), float(_vhead.base_decay)
        assert int(_vhead.in_dim) == int(_vcfg.get("in_dim", _vhead.in_dim)), \
            f"cvc head in_dim {_vhead.in_dim} != cfg {_vcfg.get('in_dim')}"
        assert all(np.isfinite([_lo, _hi, _bd])), f"cvc decay bounds not finite: {_lo},{_hi},{_bd}"
        assert _lo < _bd < _hi, f"cvc decay bounds must be lo<base<hi, got {_lo},{_bd},{_hi}"
        assert _hi <= 0.99 + 1e-9, f"cvc decay_hi {_hi} exceeds 0.99 safety cap"
        assert _lo >= 0.5, f"cvc decay_lo {_lo} < 0.5 (implausibly aggressive suppression)"
        with torch.no_grad():                                  # gate must be finite & in-band
            _g0 = _vhead(torch.zeros(1, 1, int(_vhead.in_dim))).flatten()
        assert torch.isfinite(_g0).all() and float(_g0.min()) >= _lo - 1e-6 \
            and float(_g0.max()) <= _hi + 1e-6, f"cvc head produced out-of-band gate {_g0.tolist()}"
        _vhead.eval().cuda()
        _enable_cvc(model.head.instance_bank, _vhead, _vcams, _vdets,
                    shift=shift, ngc2off=NGC2OFF, conf=args.cvc_conf)
        print(f"[CVC] enabled: {len(_vcams)} cams, {len(_vdets)} det-tiles, "
              f"decay∈[{_vhead.decay_lo},{_vhead.decay_hi}] conf>={args.cvc_conf} "
              f"head={os.path.basename(args.cvc_head)}", flush=True)

    rows = []
    t0 = time.time()
    n_det = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if not batch:
                continue
            batch = to_cuda(batch)
            if keep_cams is not None:
                ncam = batch["img"].shape[1]
                kc = torch.tensor([c for c in keep_cams if 0 <= c < ncam],
                                  device=batch["img"].device, dtype=torch.long)
                batch["img"] = batch["img"].index_select(1, kc)
                for key in ("projection_mat", "image_wh"):
                    v = batch.get(key)
                    if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == ncam:
                        batch[key] = v.index_select(1, kc)
                if idx == 0:
                    print(f"  DBG cam_keep={keep_cams} -> img cams {ncam}->{batch['img'].shape[1]}", flush=True)
                im = batch["img"]
                print(f"  DBG img tensor shape={tuple(im.shape)} mean={im.mean().item():.3f} "
                      f"std={im.std().item():.3f} min={im.min().item():.2f} max={im.max().item():.2f}",
                      flush=True)
                print(f"  DBG image_wh={batch['image_wh'][0,0].tolist()} "
                      f"proj00_row0={np.round(batch['projection_mat'][0,0,0].cpu().numpy(),3).tolist()}",
                      flush=True)
            if ccs_on:
                model.head.instance_bank.set_ccs_frame(actual_frames[idx])
            if cvc_on:
                model.head.instance_bank.set_cvc_frame(actual_frames[idx])
            out = model.simple_test(batch["img"], batch)
            res = out[0]["img_bbox"]
            boxes = res["boxes_3d"].numpy()
            scores = res["scores_3d"].numpy()
            labels = res["labels_3d"].numpy()
            if idx < 3:
                sc = np.sort(scores)[::-1]
                print(f"  DBG f{idx}: nboxes={len(scores)} top5={np.round(sc[:5],3).tolist()} "
                      f">0.1={int((scores>0.1).sum())} >0.3={int((scores>0.3).sum())} "
                      f"labels_top5={labels[np.argsort(scores)[::-1][:5]].tolist()} "
                      f"box0={np.round(boxes[np.argmax(scores),:7],2).tolist() if len(boxes) else None}", flush=True)
            ids = res["instance_ids"].numpy() if "instance_ids" in res else np.arange(len(boxes))
            fnum = actual_frames[idx]
            keep = scores >= args.score_thr
            if args.topk and keep.sum() > args.topk:
                order = np.argsort(scores)[::-1]
                sel = order[np.isin(order, np.where(keep)[0])][:args.topk]
                keep = np.zeros_like(keep); keep[sel] = True
            for b, s, lab, oid in zip(boxes[keep], scores[keep], labels[keep], ids[keep]):
                x, y, z = b[0] + shift[0], b[1] + shift[1], b[2]  # back to GT/original frame
                w, l, h = b[3], b[4], b[5]
                if args.swap_wl:
                    w, l = l, w
                yaw = b[6]
                cid = NGC2OFF.get(int(lab), int(lab)) if args.remap else int(lab)
                # 12th col = detection score (make_submission drops it; floor_roi/postproc keep it;
                # eval reads first 11). Enables per-class thresholding, NMS, gap-fill downstream.
                rows.append(f"{args.scene_id} {cid} {int(oid)} {int(fnum)} "
                            f"{x:.2f} {y:.2f} {z:.2f} {w:.2f} {l:.2f} {h:.2f} {yaw:.2f} {float(s):.4f}")
                n_det += 1
            if (idx + 1) % 50 == 0:
                dt = time.time() - t0
                print(f"  [{idx+1}/{len(ds)}] dets={n_det} {(idx+1)/dt:.2f} fps", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"wrote {len(rows)} rows -> {args.out}  ({time.time()-t0:.1f}s)", flush=True)
    if ccs_on:
        print(model.head.instance_bank.ccs_report(), flush=True)
    if rows:
        zs = np.array([float(r.split()[6]) for r in rows])
        yaws = np.array([float(r.split()[10]) for r in rows])
        print(f"z range [{zs.min():.2f},{zs.max():.2f}] mean {zs.mean():.2f} | "
              f"yaw range [{yaws.min():.2f},{yaws.max():.2f}]", flush=True)


if __name__ == "__main__":
    main()
