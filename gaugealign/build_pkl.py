"""Build an OVPKL info pickle + matched GT track1 for a warehouse scene.

Extracts frames from Camera_*.mp4 at the chosen frame numbers, writes them as
jpg, and builds the `infos` list the TAO Omniverse3DDetTrackDataset expects:
  info = {token, timestamp, scene_name, frame_idx, group_idx, scene_idx,
          cams:{cam_id:{data_path, cam_intrinsic(3x3), sensor2world_transform(4x4 world->cam)}},
          gt_boxes, gt_names, gt_velocity, valid_flag, num_lidar_pts}
Also writes a GT track1.txt for the same actual frame numbers (for eval).
"""
import os, json, argparse, pickle
import numpy as np
import cv2

# OFFICIAL Track1 class ids (verified against ZV TrackEval CLASS_NAMES).
# Official order: [person, forklift, nova_carter, transporter, gr1_t2, agility_digit, pallet_truck]
TYPE2CID = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2, "Nova_Carter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4, "Fourier_GR1_T2": 4, "GR1T2": 4, "gr1_t2": 4,
    "AgilityDigit": 5, "Agility_Digit": 5,
    "PalletTruck": 6, "pallet_truck": 6,
}
# GT object-type -> model class NAME (NGC order, matches cfg.dataset.classes)
TYPE2NAME = {
    "Person": "person",
    "FourierGR1T2": "gr1_t2", "Fourier_GR1_T2": "gr1_t2", "GR1T2": "gr1_t2",
    "AgilityDigit": "agility_digit", "Agility_Digit": "agility_digit",
    "NovaCarter": "nova_carter", "Nova_Carter": "nova_carter",
    "Transporter": "transporter",
    "Forklift": "forklift",
    "PalletTruck": "pallet_truck",
}
FPS = 30.0


def load_calib(path, shift=(0.0, 0.0), half_res=False):
    """Load calibration. `shift`=(Cx,Cy): re-center the world frame to world-shift
    so the scene content sits near the anchor origin. Predictions must add `shift`
    back to recover the original (GT) frame. `half_res`: intrinsic scaled by 0.5
    to match 960x540 decoded frames (identical model input to full-res+0.5 resize)."""
    cal = json.load(open(path))
    C = np.array([shift[0], shift[1], 0.0], dtype=np.float64)
    cams = {}
    for s in cal["sensors"]:
        if s.get("type") != "camera":
            continue
        K = np.array(s["intrinsicMatrix"], dtype=np.float64)  # 3x3
        if half_res:
            K = K.copy(); K[0, :] *= 0.5; K[1, :] *= 0.5  # scale image by 0.5
        ext = np.array(s["extrinsicMatrix"], dtype=np.float64)  # 3x4 world->cam
        R = ext[:3, :3]
        t = ext[:3, 3]
        # new_world = old_world - C  =>  cam = R*new_world + (R@C + t)
        t_new = t + R @ C
        ext4 = np.eye(4, dtype=np.float64)
        ext4[:3, :3] = R
        ext4[:3, 3] = t_new
        cams[s["id"]] = {"K": K, "ext4": ext4}
    return cams


def extract_frames(scene_dir, cam_ids, frames, out_dir, half_res=False):
    """Extract given frame numbers for each camera. Returns {cam:{frame:jpgpath}}.
    half_res: save frames resized to 960x540."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {c: {} for c in cam_ids}
    fset = sorted(set(frames))
    contiguous = len(fset) > 1 and (fset[-1] - fset[0] + 1) == len(fset)
    for cam in cam_ids:
        vid = os.path.join(scene_dir, "videos", f"{cam}.mp4")
        cap = cv2.VideoCapture(vid)
        cam_out = os.path.join(out_dir, cam)
        os.makedirs(cam_out, exist_ok=True)
        need = [f for f in fset if not os.path.exists(os.path.join(cam_out, f"{f:06d}.jpg"))]
        for f in fset:
            paths[cam][f] = os.path.join(cam_out, f"{f:06d}.jpg")
        if need:
            if contiguous:
                # sequential decode from first needed frame (fast, no per-frame seek)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fset[0])
                want = set(need)
                cur = fset[0]
                last = fset[-1]
                while cur <= last:
                    ok, img = cap.read()
                    if not ok:
                        raise RuntimeError(f"failed reading frame {cur} of {vid}")
                    if cur in want:
                        if half_res:
                            img = cv2.resize(img, (960, 540))
                        cv2.imwrite(paths[cam][cur], img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    cur += 1
            else:
                for f in need:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ok, img = cap.read()
                    if not ok:
                        raise RuntimeError(f"failed to read frame {f} of {vid}")
                    if half_res:
                        img = cv2.resize(img, (960, 540))
                    cv2.imwrite(paths[cam][f], img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cap.release()
        print(f"  {cam}: {len(fset)} frames ({len(need)} new) -> {cam_out}", flush=True)
    return paths


def _frame_gt(objs, shift):
    """Build re-centered gt arrays for one frame from ground_truth.json objects."""
    boxes, names, ids, vel = [], [], [], []
    for o in objs:
        nm = TYPE2NAME.get(o.get("object type"))
        if nm is None:
            continue
        x, y, z = o["3d location"]
        w, l, h = o["3d bounding box scale"]
        yaw = o["3d bounding box rotation"][2]
        boxes.append([x - shift[0], y - shift[1], z, w, l, h, yaw])  # re-centered
        names.append(nm)
        ids.append(int(o["object id"]))
        vel.append([0.0, 0.0, 0.0])
    n = len(boxes)
    return {
        "gt_boxes": np.array(boxes, np.float32).reshape(n, 7),
        "gt_names": np.array(names, dtype="<U32"),
        "gt_velocity": np.array(vel, np.float32).reshape(n, 3),
        "valid_flag": np.ones((n,), bool),
        "num_lidar_pts": np.ones((n,), np.int64),
        "instance_inds": np.array(ids, np.int64),
        "asset_inds": np.array(ids, np.int64),
    }


def build_infos(scene_name, calib, frame_paths, frames, shift=(0.0, 0.0), gt_json=None):
    infos = []
    cam_ids = sorted(calib.keys())
    gt = json.load(open(gt_json)) if gt_json else None
    for i, f in enumerate(frames):
        cams = {}
        for cam in cam_ids:
            cams[cam] = {
                "data_path": frame_paths[cam][f],
                "cam_intrinsic": calib[cam]["K"].astype(np.float32),
                "sensor2world_transform": calib[cam]["ext4"].astype(np.float32),
            }
        info = {
            "token": f"{scene_name}_{f:06d}",
            "timestamp": float(f) / FPS,
            "scene_name": scene_name,
            "frame_idx": i,          # 0-based within processed sequence; first == 0
            "actual_frame": int(f),  # real video frame number (for output)
            "group_idx": -1,
            "scene_idx": 0,
            "cams": cams,
        }
        if gt is not None:
            info.update(_frame_gt(gt.get(str(f), []), shift))
        else:
            info.update({
                "gt_boxes": np.zeros((0, 7), np.float32),
                "gt_names": np.array([], dtype="<U32"),
                "gt_velocity": np.zeros((0, 3), np.float32),
                "valid_flag": np.zeros((0,), bool),
                "num_lidar_pts": np.zeros((0,), np.int64),
                "instance_inds": np.zeros((0,), np.int64),
                "asset_inds": np.zeros((0,), np.int64),
            })
        infos.append(info)
    _add_velocities(infos)
    return infos


def _add_velocities(infos):
    """Codex fix 4a: GT velocity was hardcoded [0,0,0] in _frame_gt while the model REGRESSES velocity
    (objects move ~0.15-0.25 m/s). Compute real per-object velocities by CENTRAL DIFFERENCE across the
    processed frames a given track id appears in, using the info timestamps. (Endpoints use one-sided diff;
    singleton tracks stay 0.) 3D (vx,vy,vz); model uses the first components it needs. In-place on infos."""
    from collections import defaultdict
    traj = defaultdict(list)  # id -> [(info_idx, box_row_idx, timestamp, pos_xyz)]
    for ii, info in enumerate(infos):
        ids = info.get("instance_inds")
        boxes = info.get("gt_boxes")
        if ids is None or boxes is None or len(ids) == 0:
            continue
        ts = info["timestamp"]
        for jj in range(len(ids)):
            traj[int(ids[jj])].append((ii, jj, ts, boxes[jj, :3].astype(np.float64)))
    for occ in traj.values():
        occ.sort(key=lambda x: x[0])
        if len(occ) < 2:
            continue
        for k in range(len(occ)):
            ii, jj, ts, pos = occ[k]
            if k == 0:
                _, _, ts2, pos2 = occ[1]; dt = ts2 - ts; ref = pos2 - pos
            elif k == len(occ) - 1:
                _, _, ts0, pos0 = occ[k - 1]; dt = ts - ts0; ref = pos - pos0
            else:
                _, _, ts0, pos0 = occ[k - 1]; _, _, ts2, pos2 = occ[k + 1]; dt = ts2 - ts0; ref = pos2 - pos0
            if dt > 0:
                infos[ii]["gt_velocity"][jj] = (ref / dt).astype(np.float32)
    return infos


def write_gt_track1(gt_json, frames, scene_id, out_path):
    gt = json.load(open(gt_json))
    n = 0
    with open(out_path, "w") as fo:
        for f in frames:
            objs = gt.get(str(f), [])
            for o in objs:
                t = o.get("object type")
                cid = TYPE2CID.get(t)
                if cid is None:
                    continue
                oid = int(o["object id"])
                x, y, z = o["3d location"]
                w, l, h = o["3d bounding box scale"]
                yaw = o["3d bounding box rotation"][2]
                fo.write(f"{scene_id} {cid} {oid} {f} {x:.2f} {y:.2f} {z:.2f} "
                         f"{w:.2f} {l:.2f} {h:.2f} {yaw:.2f}\n")
                n += 1
    print(f"GT track1: {n} rows -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--scene_name", required=True)  # e.g. Warehouse_022
    ap.add_argument("--scene_id", type=int, required=True)  # e.g. 22
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)  # exclusive
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_pkl", required=True)
    ap.add_argument("--out_gt", default=None)
    ap.add_argument("--half_res", action="store_true",
                    help="decode frames at 960x540 + halve intrinsic (7x less disk, same model input)")
    ap.add_argument("--with_gt", action="store_true",
                    help="populate re-centered GT into infos (for training)")
    ap.add_argument("--auto_center", action="store_true",
                    help="re-center world frame to mean camera position (GT-free)")
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=0.0)
    args = ap.parse_args()

    frames = list(range(args.start, args.end, args.stride))
    print(f"{args.scene_name}: {len(frames)} frames [{frames[0]}..{frames[-1]}] stride {args.stride}", flush=True)

    shift = (args.shift_x, args.shift_y)
    if args.auto_center:
        cal = json.load(open(os.path.join(args.scene_dir, "calibration.json")))
        cc = []
        for s in cal["sensors"]:
            if s.get("type") != "camera":
                continue
            ext = np.array(s["extrinsicMatrix"]); R = ext[:3, :3]; t = ext[:3, 3]
            cc.append((-R.T @ t)[:2])
        shift = tuple(np.mean(cc, axis=0).tolist())
    print(f"world re-center shift C = {np.round(shift,2).tolist()} (predictions add this back)", flush=True)

    calib = load_calib(os.path.join(args.scene_dir, "calibration.json"), shift=shift, half_res=args.half_res)
    cam_ids = sorted(calib.keys())
    print(f"cameras: {cam_ids}  half_res={args.half_res}", flush=True)
    fp = extract_frames(args.scene_dir, cam_ids, frames, args.frames_dir, half_res=args.half_res)
    gt_json = os.path.join(args.scene_dir, "ground_truth.json") if args.with_gt else None
    infos = build_infos(args.scene_name, calib, fp, frames, shift=shift, gt_json=gt_json)
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump({"infos": infos,
                     "metadata": {"version": args.scene_name,
                                  "recenter_shift": list(shift)}}, f)
    print(f"wrote {len(infos)} infos -> {args.out_pkl}", flush=True)
    if args.out_gt:
        gtj = os.path.join(args.scene_dir, "ground_truth.json")
        if os.path.exists(gtj):
            write_gt_track1(gtj, frames, args.scene_id, args.out_gt)


if __name__ == "__main__":
    main()
