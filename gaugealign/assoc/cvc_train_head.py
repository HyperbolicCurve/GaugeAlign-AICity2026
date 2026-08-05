"""cvc_train_head.py — offline trainer for the CVC-Assoc survival gate (our architecture).

Two modes:

  make-init   Construct ONE CVCAssocHead with a fixed seed and save its state_dict to
              --out. This is the canonical init head. It MUST be reused (via --head_init)
              by BOTH cvc_label_targets.py (feature dump) AND inference, because
              build_feature bakes id_proj(feat_t) into the dumped X — id_proj is a FROZEN
              random projection in v1, so every stage has to see the same weights or the
              59-d feature vectors are inconsistent. (Training id_proj is a v2 ablation that
              would require dumping the raw 256-d feat_t instead.)

  train       Load the init head, load one or more label npz (X[N,in_dim], Y[N] survival),
              freeze id_proj, train ONLY the MLP with class-weighted BCE + a floor-anchor
              regularizer, and save the trained head to --out ONLY IF it beats the scalar
              floor on scene-disjoint held-out data. Floor-safety is preserved end-to-end:
              the init head's final layer gives g==0.90 bit-exact; training only moves it
              where labels support it, the (base±amp) tanh squash bounds every gate to
              ~[0.81,0.99], and refuse-to-save keeps the proven floor when generalization fails.

The gate value is g = base_decay + amp*tanh(mlp(x)) (amp=min(base-lo,hi-base)); we train
mlp(x) so sigmoid(mlp(x)) predicts the survival bit y (1=persistent claimed match, 0=decay),
so high-survival tracks decay slowly (g->hi) and stale/duplicate tracks decay fast (g->lo).

Runs on CPU in seconds. Example:
  python cvc_train_head.py make-init --out cvc_head_init.pt
  python cvc_label_targets.py ... --head_init cvc_head_init.pt --out cvc_labels/w000.npz
  python cvc_train_head.py train --head_init cvc_head_init.pt \
      --npz cvc_labels/w000.npz cvc_labels/w002.npz --out cvc_head_trained.pt
"""
import os, sys, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvc_assoc_head import CVCAssocHead


def head_id_hash(head):
    """sha1[:16] of the FROZEN id_proj weights baked into every dumped X. Must agree between the
    make-init head, every label npz, and the trained head (fix F/N provenance)."""
    h = hashlib.sha1()
    sd = head.state_dict()
    for k in ("id_proj.weight", "id_proj.bias"):
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def build_head(args):
    return CVCAssocHead(embed_dims=args.embed_dims, id_proj_dim=args.id_proj_dim,
                        hidden=args.hidden, decay_lo=args.decay_lo,
                        decay_hi=args.decay_hi, base_decay=args.base_decay)


def save_head(head, path):
    torch.save({
        "state_dict": head.state_dict(),
        "cfg": dict(embed_dims=head.id_proj.in_features, id_proj_dim=head.id_proj.out_features,
                    hidden=head.mlp[0].out_features, decay_lo=head.decay_lo,
                    decay_hi=head.decay_hi, in_dim=head.in_dim),
    }, path)


def load_head(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["cfg"]
    head = CVCAssocHead(embed_dims=cfg["embed_dims"], id_proj_dim=cfg["id_proj_dim"],
                        hidden=cfg["hidden"], decay_lo=cfg["decay_lo"], decay_hi=cfg["decay_hi"])
    head.load_state_dict(ckpt["state_dict"])
    return head, cfg


def make_init(args):
    torch.manual_seed(args.seed)
    head = build_head(args)
    # BIT-EXACT floor-safe init: g must equal float32(base_decay) EVERYWHERE, not merely close.
    # A ~1e-8 offset can flip a top-k tie at inference and drift off the proven scalar floor, so
    # we require exact equality (the head's tanh reparam + zero weight AND bias guarantees it).
    head.eval()
    with torch.no_grad():
        g = head(torch.zeros(1, 1, head.in_dim))
    expected = torch.full_like(g, float(args.base_decay))     # float32(base_decay)
    if not torch.equal(g, expected):
        raise AssertionError(f"floor-safe init NOT bit-exact: g={g.flatten().tolist()} "
                             f"!= base_decay={args.base_decay}")
    save_head(head, args.out)
    print(f"[make-init] saved init head -> {args.out}  in_dim={head.in_dim} "
          f"g_init={g.flatten()[0].item():.8f} (bit-exact base) "
          f"decay=[{head.decay_lo},{head.decay_hi}] id_proj_hash={head_id_hash(head)} "
          f"seed={args.seed}")


def _auc(scores, labels):
    """Rank-based ROC-AUC (no sklearn dependency)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def train(args):
    head, cfg = load_head(args.head_init)
    init_hash = head_id_hash(head)
    # freeze id_proj (baked into X at label-time) -> only the MLP trains
    for p in head.id_proj.parameters():
        p.requires_grad_(False)

    # ---- load npz with STRICT provenance checks (fix F/N) ----
    Xs, Ys, Ss = [], [], []
    for f in args.npz:
        d = np.load(f)
        if int(d["in_dim"]) != head.in_dim:
            raise ValueError(f"{f}: in_dim {int(d['in_dim'])} != head {head.in_dim}")
        hh = str(d["head_hash"]) if "head_hash" in d else None
        if hh is None:
            raise ValueError(f"{f}: missing head_hash — re-run the labeler (provenance required).")
        if hh != init_hash:
            raise ValueError(f"{f}: id_proj hash {hh} != --head_init {init_hash}. These features "
                             f"were dumped with a DIFFERENT frozen id_proj; mixing them corrupts X.")
        if "has_consensus" in d and int(d["has_consensus"]) != 1:
            raise ValueError(f"{f}: has_consensus=0 (dry-run/zeros features) — not trainable.")
        meta = d["meta"]                       # [N,4] = (scene, frame, slot, cls)
        Xs.append(d["X"].astype(np.float32))
        Ys.append(d["Y"].astype(np.float32))
        Ss.append(meta[:, 0].astype(np.int64))
    X = torch.from_numpy(np.concatenate(Xs, 0))
    Y = torch.from_numpy(np.concatenate(Ys, 0))
    S = np.concatenate(Ss, 0)                   # per-row scene id

    # ---- scene-DISJOINT held-out split (generalization, not train-fit) ----
    uniq = sorted(set(int(s) for s in S))
    if args.val_scenes:
        val_scenes = set(int(s) for s in args.val_scenes)
    elif len(uniq) >= 2:
        n_val = max(1, round(0.2 * len(uniq)))
        val_scenes = set(uniq[-n_val:])        # hold out the highest scene ids
    else:
        val_scenes = set()                     # single scene: no disjoint holdout possible
    val_mask = np.isin(S, list(val_scenes)) if val_scenes else np.zeros(len(S), bool)
    tr_mask = ~val_mask
    Xtr, Ytr = X[tr_mask], Y[tr_mask]
    Xva, Yva = X[val_mask], Y[val_mask]
    n_pos = float((Ytr == 1).sum()); n_neg = float((Ytr == 0).sum())
    print(f"[train] {len(Y)} samples  train={int(tr_mask.sum())} val={int(val_mask.sum())} "
          f"(val_scenes={sorted(val_scenes) or 'NONE — single scene, holdout AUC unreliable'})")
    print(f"[train] train pos={int(n_pos)} neg={int(n_neg)} "
          f"({100*n_pos/max(len(Ytr),1):.1f}% pos)  in_dim={head.in_dim}  id_proj_hash={init_hash}")
    if n_pos == 0 or n_neg == 0:
        print("[train] REFUSE: single-class train labels — keeping scalar floor (not saving).")
        return
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])

    head.eval()
    with torch.no_grad():
        base_auc_tr = _auc(head.mlp(Xtr).squeeze(-1).numpy(), Ytr.numpy())
        base_auc_va = (_auc(head.mlp(Xva).squeeze(-1).numpy(), Yva.numpy())
                       if val_mask.any() else float("nan"))
    print(f"[train] init AUC train={base_auc_tr:.4f} val={base_auc_va:.4f} (floor gate g==0.90)")

    opt = torch.optim.Adam([p for p in head.mlp.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    base_g = head._base.item()
    N = len(Ytr); idx = np.arange(N)
    head.train()
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        rng.shuffle(idx)
        tot = tot_bce = tot_reg = 0.0
        for s in range(0, N, args.batch_size):
            b = idx[s:s + args.batch_size]
            xb = Xtr[b]; yb = Ytr[b]
            logit = head.mlp(xb).squeeze(-1)
            bce = lossf(logit, yb)
            # FLOOR-ANCHOR regularizer: pull the gate toward base_decay unless the labels clearly
            # justify moving it. g = base + amp*tanh(logit); penalize (g-base)^2 so the head stays
            # near the proven scalar floor by default (bounds divergence risk below the floor).
            g = head._base + head._amp * torch.tanh(logit)
            reg = ((g - base_g) ** 2).mean()
            loss = bce + args.floor_lambda * reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b); tot_bce += float(bce) * len(b); tot_reg += float(reg) * len(b)
        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                lg_tr = head.mlp(Xtr).squeeze(-1)
                auc_va = (_auc(head.mlp(Xva).squeeze(-1).numpy(), Yva.numpy())
                          if val_mask.any() else float("nan"))
                g = head(Xtr.unsqueeze(1)).squeeze(1)
            print(f"  ep{ep+1:3d} loss={tot/N:.4f} bce={tot_bce/N:.4f} reg={tot_reg/N:.5f} "
                  f"AUC train={_auc(lg_tr.numpy(), Ytr.numpy()):.4f} val={auc_va:.4f} "
                  f"g[min/mean/max]={g.min():.3f}/{g.mean():.3f}/{g.max():.3f}")
            head.train()

    head.eval()
    with torch.no_grad():
        val_auc = (_auc(head.mlp(Xva).squeeze(-1).numpy(), Yva.numpy())
                   if val_mask.any() else float("nan"))
        g_all = head(X.unsqueeze(1)).squeeze(1)
        gp = g_all[Y == 1].mean(); gn = g_all[Y == 0].mean()

    # ---- REFUSE-TO-SAVE if the learned gate does not beat the scalar floor on held-out scenes ----
    # The scalar floor is a constant gate (no discrimination => AUC 0.5). Saving a head that does
    # not generalize would risk regressing below the proven floor, so we keep the floor instead.
    if val_mask.any():
        if not (val_auc >= args.min_val_auc):
            print(f"[train] REFUSE: held-out AUC {val_auc:.4f} < min_val_auc {args.min_val_auc} "
                  f"— gate does not beat the scalar floor on unseen scenes. NOT saving; the "
                  f"byte-identical floor path stays armed.")
            return
    else:
        print("[train] WARNING: no scene-disjoint holdout (single scene). Cannot certify "
              "generalization; save gated on --allow_no_holdout.")
        if not args.allow_no_holdout:
            print("[train] REFUSE: pass --allow_no_holdout to save without a held-out AUC gate.")
            return

    save_head(head, args.out)
    print(f"[train] SAVED -> {args.out}  val_AUC={val_auc:.4f} (init {base_auc_va:.4f})  "
          f"g(pos)={gp:.3f} g(neg)={gn:.3f} separation={float(gp-gn):+.3f}  "
          f"id_proj_hash={head_id_hash(head)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    mi = sub.add_parser("make-init")
    mi.add_argument("--out", required=True)
    mi.add_argument("--embed_dims", type=int, default=256)
    mi.add_argument("--id_proj_dim", type=int, default=32)
    mi.add_argument("--hidden", type=int, default=128)
    mi.add_argument("--decay_lo", type=float, default=0.80)
    mi.add_argument("--decay_hi", type=float, default=0.99)
    mi.add_argument("--base_decay", type=float, default=0.90)
    mi.add_argument("--seed", type=int, default=0)
    mi.set_defaults(func=make_init)

    tr = sub.add_parser("train")
    tr.add_argument("--head_init", required=True, help="cvc_head_init.pt from make-init")
    tr.add_argument("--npz", nargs="+", required=True, help="label npz files")
    tr.add_argument("--out", required=True)
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch_size", type=int, default=4096)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--val_scenes", nargs="*", type=int, default=None,
                    help="scene ids to hold out for the generalization-AUC gate (default: top 20%%)")
    tr.add_argument("--floor_lambda", type=float, default=0.05,
                    help="floor-anchor reg weight: penalize (g-base_decay)^2 to stay near the floor")
    tr.add_argument("--min_val_auc", type=float, default=0.55,
                    help="refuse to save unless held-out AUC >= this (scalar floor == 0.5)")
    tr.add_argument("--allow_no_holdout", action="store_true",
                    help="permit saving when only one scene is present (no disjoint holdout)")
    tr.set_defaults(func=train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
