#!/usr/bin/env python3
"""Validate + package a Track 1 submission (track1.txt -> track1.zip).

Official format (per 2026 rules): one line per detected object, 11 space-separated fields
  <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
- scene_id, class_id (0..6), object_id (positive, unique per scene+class), frame_id: ints
- x y z w l h yaw: floats ROUNDED TO 2 DECIMALS
- archive as track1.zip, <= 50 MB

Accepts a raw pred file that is 11-col OR 12-col (drops a trailing score column),
enforces the format, drops invalid rows (with a report), writes the clean txt + zip.
"""
from __future__ import annotations
import argparse, os, zipfile
from collections import Counter

VALID_CLASSES = set(range(7))  # Person..PalletTruck = 0..6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="raw track1 prediction file (11 or 12 col)")
    ap.add_argument("--out_txt", default="track1.txt")
    ap.add_argument("--out_zip", default="track1.zip")
    ap.add_argument("--strict", action="store_true", help="error (not drop) on any bad row")
    args = ap.parse_args()

    rows, dropped = [], 0
    scenes, classes = Counter(), Counter()
    idmap = {}          # (scene,class,orig_oid) -> new positive oid (dense per scene+class)
    next_oid = {}       # (scene,class) -> next id to assign, starting at 1
    with open(args.pred) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 11:
                dropped += 1
                if args.strict:
                    raise SystemExit(f"line {ln}: <11 fields: {line!r}")
                continue
            try:
                sid = int(float(p[0])); cid = int(float(p[1]))
                oid = int(float(p[2])); fid = int(float(p[3]))
                vals = [float(x) for x in p[4:11]]  # x y z w l h yaw
            except ValueError:
                dropped += 1
                if args.strict:
                    raise SystemExit(f"line {ln}: non-numeric: {line!r}")
                continue
            if cid not in VALID_CLASSES:
                dropped += 1
                continue
            # Remap object_id to a POSITIVE (>=1), track-consistent, per-(scene,class) id.
            key = (sid, cid, oid)
            if key not in idmap:
                sc = (sid, cid)
                next_oid[sc] = next_oid.get(sc, 0) + 1
                idmap[key] = next_oid[sc]
            new_oid = idmap[key]
            scenes[sid] += 1; classes[cid] += 1
            rows.append(f"{sid} {cid} {new_oid} {fid} " + " ".join(f"{v:.2f}" for v in vals))
    kept = rows

    with open(args.out_txt, "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.out_txt, arcname="track1.txt")

    txt_mb = os.path.getsize(args.out_txt) / 1e6
    zip_mb = os.path.getsize(args.out_zip) / 1e6
    print(f"rows kept: {len(kept)}  dropped: {dropped}")
    print(f"scenes: {dict(sorted(scenes.items()))}")
    print(f"classes (0=Person..6=PalletTruck): {dict(sorted(classes.items()))}")
    print(f"{args.out_txt}: {txt_mb:.2f} MB  |  {args.out_zip}: {zip_mb:.2f} MB  (limit 50 MB)")
    assert zip_mb <= 50, "ZIP EXCEEDS 50 MB LIMIT"
    if not kept:
        print("WARNING: submission is EMPTY")
    print("OK: submission packaged ->", args.out_zip)


if __name__ == "__main__":
    main()
