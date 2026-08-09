"""
Heterogeneous GNN-VAE pipeline — Stage-B cross-dataset adapter alignment.

Transfer ablation ("SEAN-only -> SNSB downstream" and the symmetric direction).

Given a Stage-A checkpoint that was pretrained on ONE dataset only
(train_unified_gnn_vae_v3.py with --snsb_loss_weight 0  or  --sean_loss_weight 0),
the held-out dataset's front-end adapter is still at initialization because it
never received a gradient. This script trains ONLY that adapter so the held-out
dataset can be projected into the FROZEN, single-dataset latent geometry.

What is trainable vs frozen (for --align snsb):
    TRAINABLE : encoder.snsb_robot_stem, encoder.snsb_bystander_stem,
                encoder.snsb_robot_embed, encoder.snsb_bystander_embed,
                encoder.edge_encoders["snsb"], decoder.edge_decoders["snsb"]
    FROZEN    : everything else — the shared trunk (mp_layers, fc_mu, fc_logvar,
                encoder/decoder edge_type_embed, decoder.pre_decode) AND the
                already-trained SEAN branch.

Symmetric for --align sean (SEAN stems/edge-encoder + sean_robot_head +
edge_decoders["sean"] trainable; everything else frozen).

Loss is identical to the main VAE trainer's compute_loss for that dataset:
    SNSB : weighted edge-recon + beta * KL   (no robot-recon target)
    SEAN : robot-recon + weighted edge-recon + beta * KL
Message passing is NOT skipped — data flows through the frozen trunk and
gradients backprop through it into the adapter. Adapter training is fully
self-supervised (reconstruction only; no labels), and only on that dataset's
TRAIN split (val used for early stopping; oracle/test never touched here).

The saved checkpoint mirrors train_unified_gnn_vae_v3.py's schema exactly, so
the existing probes / eval_label_efficiency.py load it with no changes.

Usage (run from src/):
    # Dir 1: SEAN-only pretrain -> align the SNSB adapter (for SNSB downstream)
    CUDA_VISIBLE_DEVICES=0 python train_stage_b_adapter_align.py \
        --align snsb \
        --stage_a_ckpt checkpoints/transfer_seanonly/best_unified_gnn_vae_v3_seednone.pt \
        --out_dir      checkpoints/transfer_seanonly_alignSNSB

    # Dir 2: SNSB-only pretrain -> align the SEAN adapter (for SEAN downstream)
    CUDA_VISIBLE_DEVICES=1 python train_stage_b_adapter_align.py \
        --align sean \
        --stage_a_ckpt checkpoints/transfer_snsbonly/best_unified_gnn_vae_v3_seednone.pt \
        --out_dir      checkpoints/transfer_snsbonly_alignSEAN
"""

import argparse
import os
import shutil
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import UnifiedGNNVAE

# Reuse the exact helpers / constants from the main trainer (no duplication).
from train_unified_gnn_vae_v3 import (
    SEAN_ID,
    SEANHeteroWrapper,
    DropSNSBNodeFeatures,
    compute_loss,
    get_snsb_split_indices,
    make_loader,
    run_eval,
)


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def configure_trainable(model, align):
    """Freeze the whole model, then unfreeze ONLY the align branch's adapter + decoder head."""
    for p in model.parameters():
        p.requires_grad = False

    enc = model.encoder
    dec = model.decoder

    if align == "snsb":
        _set_requires_grad(enc.snsb_robot_stem, True)
        _set_requires_grad(enc.snsb_bystander_stem, True)
        _set_requires_grad(enc.edge_encoders["snsb"], True)
        _set_requires_grad(dec.edge_decoders["snsb"], True)
        enc.snsb_robot_embed.requires_grad = True
        enc.snsb_bystander_embed.requires_grad = True
    elif align == "sean":
        _set_requires_grad(enc.sean_robot_stem, True)
        _set_requires_grad(enc.sean_bystander_stem, True)
        _set_requires_grad(enc.edge_encoders["sean"], True)
        _set_requires_grad(dec.sean_robot_head, True)
        _set_requires_grad(dec.edge_decoders["sean"], True)
        enc.sean_bystander_embed.requires_grad = True
    else:
        raise ValueError(f"--align must be 'sean' or 'snsb', got {align}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Stage B] align={align}  trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / max(1, n_total):.2f}%)")
    return trainable


def build_align_loaders(args, ckpt):
    """Return (train_loader, val_loader) for the single dataset being aligned."""
    h5_path    = args.h5_path      or ckpt.get("h5_path", "dataloaders/combined_v1.h5")
    snsb_h5    = args.snsb_h5_path or ckpt.get("snsb_h5_path", "dataloaders/social_nav_v3.h5")
    split_file = args.split_file   or ckpt.get("split_file", "checkpoints/split_indices_final.pt")

    combined_ds = UnifiedGraphDataset(h5_path, ssub=False)
    split = torch.load(split_file, map_location="cpu", weights_only=False)

    # Reuse the exact train-split normalization stats saved by Stage A (no leakage,
    # identical to pretraining).
    combined_ds.robot_mean = ckpt.get("sean_robot_mean")
    combined_ds.robot_std  = ckpt.get("sean_robot_std")

    if args.align == "sean":
        sean_train = [i for i in split["train"] if combined_ds._dataset_ids[i] == SEAN_ID]
        sean_val   = [i for i in split["val"]   if combined_ds._dataset_ids[i] == SEAN_ID]
        if combined_ds.robot_mean is None or combined_ds.robot_std is None:
            combined_ds.set_robot_stats(sean_train)
        print(f"[Stage B] SEAN align  train={len(sean_train)}  val={len(sean_val)}")
        train_sub = SEANHeteroWrapper(torch.utils.data.Subset(combined_ds, sean_train))
        val_sub   = SEANHeteroWrapper(torch.utils.data.Subset(combined_ds, sean_val))
    else:  # align == "snsb"
        snsb_ds = UnifiedSocialNavDataset(snsb_h5, zero_node_features=True)
        snsb_train_idx, snsb_val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)
        print(f"[Stage B] SNSB align  train={len(snsb_train_idx)}  val={len(snsb_val_idx)}")
        train_sub = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_ds, snsb_train_idx))
        val_sub   = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_ds, snsb_val_idx))

    # Plain shuffle, no weighted/stratified sampler: adapter fit is unsupervised.
    train_loader = make_loader(train_sub, args.batch_size, True,  args.num_workers)
    val_loader   = make_loader(val_sub,   args.batch_size, False, args.num_workers)
    return train_loader, val_loader


def resolve_split_used(ckpt, stage_a_ckpt_path, split_file_arg=None):
    """Locate the split_used_*.pt Stage A saved, so it can be copied next to the Stage-B ckpt.

    Checks, in order: the ckpt's saved split_path (as-is and by basename in the Stage-A
    dir), any split_used_v3_seed*.pt in the Stage-A dir, and an explicit --split_file.
    Returns the first path that exists, else None.
    """
    a_dir = os.path.dirname(os.path.abspath(stage_a_ckpt_path))
    candidates = []
    sp = ckpt.get("split_path")
    if sp:
        candidates.append(sp)                                     # as saved (may be absolute)
        candidates.append(os.path.join(a_dir, os.path.basename(sp)))  # basename in Stage-A dir
    if os.path.isdir(a_dir):
        for name in sorted(os.listdir(a_dir)):                    # any split in the Stage-A dir
            if name.startswith("split_used_v3_seed") and name.endswith(".pt"):
                candidates.append(os.path.join(a_dir, name))
    if split_file_arg:
        candidates.append(split_file_arg)                         # explicit override
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Stage B] device={device}  align={args.align}")

    ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    sa = ckpt.get("args", {}) or {}

    model = UnifiedGNNVAE(
        hidden_dim=int(sa.get("hidden_dim", 256)),
        latent_dim=int(sa.get("latent_dim", 128)),
        num_mp_layers=int(sa.get("num_mp_layers", 2)),
        dropout=float(sa.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[Stage B] loaded Stage-A checkpoint: {args.stage_a_ckpt} "
          f"(epoch={ckpt.get('epoch','NA')}, val_total={ckpt.get('val_total','NA')})")

    trainable = configure_trainable(model, args.align)

    train_loader, val_loader = build_align_loaders(args, ckpt)

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy the Stage-A split UP FRONT (fail-fast). The downstream probe/eval require
    # split_used_v3_seednone.pt in the output dir; doing this before training guarantees
    # it is present even if training is interrupted (fixes the standalone-run bug where a
    # checkpoint could be saved without its split and get silently skipped at eval time).
    split_src = resolve_split_used(ckpt, args.stage_a_ckpt, args.split_file)
    if split_src is None:
        raise FileNotFoundError(
            f"Could not locate a split_used_*.pt for Stage-A checkpoint {args.stage_a_ckpt}. "
            "Pass --split_file explicitly. Downstream eval_label_efficiency.py requires this file."
        )
    split_dst = os.path.join(args.out_dir, "split_used_v3_seednone.pt")
    shutil.copyfile(split_src, split_dst)
    print(f"[Stage B] copied split -> {split_dst}  (from {split_src})")

    best_path = os.path.join(args.out_dir, "best_unified_gnn_vae_v3_seednone.pt")
    best_val = float("inf")
    es_counter = 0

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / max(1, args.warmup_epochs))

        model.train()
        run_sum, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
            loss, r_robot, r_edge, kl = compute_loss(
                robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            run_sum += float(loss); n += 1
        train_loss = run_sum / max(1, n)

        val_m = run_eval(model, val_loader, device, beta)
        val_total = val_m["total"]
        scheduler.step(val_total)

        print(f"[Stage B] epoch {epoch:03d}/{args.epochs} | beta={beta:.4f} | "
              f"train={train_loss:.4f} | val={val_total:.4f} "
              f"(r_robot={val_m['r_robot']:.3f} r_edge={val_m['r_edge']:.3f} kl={val_m['kl']:.3f})")

        if val_total < best_val:
            best_val = val_total
            es_counter = 0
            out = dict(ckpt)                       # preserve Stage-A schema exactly
            out["model_state_dict"] = model.state_dict()
            out["epoch"] = epoch
            out["val_total"] = val_total
            out["stage_b_align"] = args.align
            out["stage_a_ckpt"] = os.path.abspath(args.stage_a_ckpt)
            torch.save(out, best_path)
            print(f"[Stage B]   saved best -> {best_path}  (val={val_total:.4f})")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"[Stage B]   early stopping at epoch {epoch} (patience={args.es_patience})")
                break

    print(f"[Stage B] done. best val={best_val:.4f}  checkpoint={best_path}  split={split_dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--align", choices=["sean", "snsb"], required=True,
                   help="Which dataset's front-end adapter to train into the frozen trunk.")
    p.add_argument("--stage_a_ckpt", required=True,
                   help="Single-dataset Stage-A checkpoint (best_unified_gnn_vae_v3_seed*.pt).")
    p.add_argument("--out_dir", required=True,
                   help="Output dir; receives best_unified_gnn_vae_v3_seednone.pt + split_used_v3_seednone.pt.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    # Defaults mirror the STARS-He pretraining config (run_train_vae.sh 256x128).
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--es_patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00011564)
    p.add_argument("--beta_max", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=25)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=20.0)
    # Data sources (default: read from the Stage-A checkpoint, else standard paths).
    p.add_argument("--h5_path", default=None, help="SEAN combined_v1.h5 (default: from ckpt).")
    p.add_argument("--snsb_h5_path", default=None, help="SNSB social_nav_v3.h5 (default: from ckpt).")
    p.add_argument("--split_file", default=None, help="split_indices_final.pt (default: from ckpt).")
    args = p.parse_args()
    train(args)
