"""
Heterogeneous GNN-VAE pipeline — combined SEAN+SNSB VAE trainer.

Train UnifiedGNNVAE v1 — SEAN from combined_v1.h5, SNSB from social_nav_v3.h5 (T=20 native).

v3 changes over v2:
  - SEAN: loaded from combined_v1.h5, split indices taken directly from split_indices_final.pt
  - SNSB: loaded from social_nav_v3.h5 at native T=20 (no interpolation).
          Scene split extracted from split_indices_final.pt via combined_v1.h5 scene attrs,
          then matched to social_nav_v3.h5 samples — identical scenes to the fixed split.
          SNSB 60D trajectory node features are dropped before the encoder; nodes are
          initialized from learned SNSB type embeddings plus incident temporal edge geometry.
  - homo_to_hetero only for SEAN (SNSB comes from UnifiedSocialNavDataset already HeteroData)
  - Standard VAE loss: recon_robot + weighted recon_edges + beta*KL, computed per dataset

Usage (5 seeds on GPUs 1-5):
  for i in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i python train_unified_gnn_vae_v3.py --seed $i --gpu 0 &
  done
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.unified_graph_dataset_v1 import (
    DATASET_SEAN,
    DATASET_SSUB_REAL,
    UnifiedGraphDataset,
)
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import (
    EDGE_TYPES as UNIFIED_EDGE_TYPES,
    NODE_TYPES,
    UnifiedGNNVAE,
)

SEAN_ID = DATASET_SEAN       # 0
SNSB_ID = DATASET_SSUB_REAL  # 1

EDGE_LOSS_WEIGHTS = {
    "sean": {
        ("robot", "to", "bystander"): 3.5,
        ("bystander", "to", "robot"): 3.5,
        ("bystander", "to", "bystander"): 0.1,
    },
    "snsb": {
        ("robot", "to", "bystander"): 3.0,
        ("bystander", "to", "robot"): 3.0,
        ("bystander", "to", "bystander"): 0.5,
    },
}

DATASET_ID_TO_KEY = {SEAN_ID: "sean", SNSB_ID: "snsb"}


# ---------------------------------------------------------------------------
# homo_to_hetero — SEAN only (combined_v1.h5 homogeneous Data → HeteroData)
# ---------------------------------------------------------------------------

def homo_to_hetero(data):
    """Convert combined_v1.h5 SEAN homogeneous Data → HeteroData for UnifiedGNNVAE."""
    node_type = data.node_type
    robot_mask = (node_type == 0)
    bystander_mask = (node_type == 1)

    n_total = node_type.shape[0]
    n_robots = int(robot_mask.sum())
    n_bystanders = int(bystander_mask.sum())

    local_idx = torch.full((n_total,), -1, dtype=torch.long)
    local_idx[robot_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_robots, dtype=torch.long)
    local_idx[bystander_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_bystanders, dtype=torch.long)

    hdata = HeteroData()
    hdata.dataset_id = torch.tensor([SEAN_ID], dtype=torch.long)
    hdata["robot"].x = data.x[robot_mask]           # [1, 592] normalized
    hdata["bystander"].num_nodes = n_bystanders      # model uses learnable embed

    if hasattr(data, "action_label"):
        hdata["bystander"].y = data.action_label[bystander_mask].long()

    src_nodes = data.edge_index[0]
    dst_nodes = data.edge_index[1]
    src_types = node_type[src_nodes]
    dst_types = node_type[dst_nodes]

    for (st, dt), et in [
        ((0, 1), ("robot", "to", "bystander")),
        ((1, 0), ("bystander", "to", "robot")),
        ((1, 1), ("bystander", "to", "bystander")),
    ]:
        mask = (src_types == st) & (dst_types == dt)
        if mask.any():
            hdata[et].edge_index = torch.stack(
                [local_idx[src_nodes[mask]], local_idx[dst_nodes[mask]]], dim=0
            )
            hdata[et].edge_attr = data.edge_attr[mask]   # [E, 40, 8]

    return hdata


class SEANHeteroWrapper(Dataset):
    """Wraps a Subset of UnifiedGraphDataset (SEAN only) and applies homo_to_hetero."""
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def len(self):
        return len(self.subset)

    def get(self, idx):
        return homo_to_hetero(self.subset[idx])


class DropSNSBNodeFeatures(Dataset):
    """Drop SNSB 60D node features so the model uses learned node-type embeddings."""
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def len(self):
        return len(self.subset)

    def get(self, idx):
        data = self.subset[idx]
        data["robot"].num_nodes = int(data["robot"].x.shape[0])
        data["bystander"].num_nodes = int(data["bystander"].x.shape[0])
        data["robot"].x = None
        data["bystander"].x = None
        return data


# ---------------------------------------------------------------------------
# SNSB scene-based split: match scenes from combined_v1.h5
# ---------------------------------------------------------------------------

def get_snsb_split_indices(combined_ds, split, snsb_ds):
    """
    Extract SNSB train/val scene sets from combined_v1.h5 + split_indices_final.pt,
    then return matching indices into social_nav_v3.h5 (real portion, N<3052).

    Returns: (snsb_train_idx, snsb_val_idx) as lists of indices into snsb_ds
    """
    N_REAL = 3052

    snsb_train_combined = [i for i in split["train"] if combined_ds._dataset_ids[i] == SNSB_ID]
    snsb_val_combined   = [i for i in split["val"]   if combined_ds._dataset_ids[i] == SNSB_ID]

    train_scenes = set(combined_ds._scenes[i] for i in snsb_train_combined)
    val_scenes   = set(combined_ds._scenes[i] for i in snsb_val_combined)

    assert not (train_scenes & val_scenes), "Scene overlap between train and val!"

    snsb_train_idx, snsb_val_idx, unmatched = [], [], []
    for i in range(N_REAL):
        sample = snsb_ds.base_dataset[i]
        scene = getattr(sample, "scene", None) or ""
        if scene in train_scenes:
            snsb_train_idx.append(i)
        elif scene in val_scenes:
            snsb_val_idx.append(i)
        else:
            unmatched.append(i)

    assert len(snsb_train_idx) == len(snsb_train_combined), (
        f"SNSB train count mismatch: combined={len(snsb_train_combined)} "
        f"social_nav_v3={len(snsb_train_idx)}"
    )
    assert len(snsb_val_idx) == len(snsb_val_combined), (
        f"SNSB val count mismatch: combined={len(snsb_val_combined)} "
        f"social_nav_v3={len(snsb_val_idx)}"
    )

    print(f"SNSB scene split verified: train={len(snsb_train_idx)} val={len(snsb_val_idx)} "
          f"unmatched={len(unmatched)} (oracle/test, expected)")

    return snsb_train_idx, snsb_val_idx


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def dataset_key_from_batch(batch):
    did = batch.dataset_id
    if torch.is_tensor(did):
        did = int(torch.unique(did.view(-1).long()).item())
    return DATASET_ID_TO_KEY[int(did)]


def compute_loss(robot_recon, edge_recons, data, mu_dict, logvar_dict, beta):
    dataset_key = dataset_key_from_batch(data)

    if robot_recon is not None:  # SEAN only
        recon_robot = F.mse_loss(robot_recon, data["robot"].x)
        device = robot_recon.device
    else:  # SNSB: no robot reconstruction target
        device = next(v.device for v in mu_dict.values() if v.numel() > 0)
        recon_robot = torch.tensor(0.0, device=device)

    edge_losses, edge_weights = [], []
    for et in UNIFIED_EDGE_TYPES:
        if et in edge_recons and hasattr(data[et], "edge_attr"):
            edge_losses.append(F.mse_loss(edge_recons[et], data[et].edge_attr))
            edge_weights.append(EDGE_LOSS_WEIGHTS[dataset_key][et])

    if edge_losses:
        w = torch.tensor(edge_weights, device=device)
        recon_edges = (w * torch.stack(edge_losses)).sum() / w.sum()
    else:
        recon_edges = torch.tensor(0.0, device=device)

    all_mu = torch.cat([mu_dict[t] for t in NODE_TYPES if mu_dict[t].shape[0] > 0], dim=0)
    all_lv = torch.cat([logvar_dict[t] for t in NODE_TYPES if logvar_dict[t].shape[0] > 0], dim=0)
    kl = -0.5 * torch.mean(1 + all_lv - all_mu.pow(2) - all_lv.exp())

    total = recon_robot + recon_edges + beta * kl
    return total, recon_robot, recon_edges, kl


def run_eval(model, loader, device, beta):
    model.eval()
    sums = {"total": 0.0, "r_robot": 0.0, "r_edge": 0.0, "kl": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
            total, r_robot, r_edge, kl = compute_loss(
                robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta
            )
            sums["total"] += float(total); sums["r_robot"] += float(r_robot)
            sums["r_edge"] += float(r_edge); sums["kl"] += float(kl)
            n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


def compute_per_sample_losses(model, dataset, device, batch_size, num_workers, dataset_key):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    was_training = model.training
    model.eval()
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
            bsz = int(batch["robot"].batch.max()) + 1 if hasattr(batch["robot"], "batch") else 1
            per_graph = torch.zeros(bsz, device=device)
            edge_w_sum = torch.zeros(bsz, device=device)
            if robot_recon is not None:
                robot_batch = batch["robot"].batch
                per_robot = F.mse_loss(robot_recon, batch["robot"].x, reduction="none").mean(dim=-1)
                per_graph.scatter_add_(0, robot_batch, per_robot)
            for et in UNIFIED_EDGE_TYPES:
                if et not in edge_recons or not hasattr(batch[et], "edge_attr"):
                    continue
                edge_index = batch[et].edge_index
                if edge_index.shape[1] == 0:
                    continue
                w = float(EDGE_LOSS_WEIGHTS[dataset_key].get(et, 1.0))
                per_edge = F.mse_loss(edge_recons[et], batch[et].edge_attr, reduction="none").mean(dim=(-2, -1))
                src_type = et[0]
                edge_batch = batch[src_type].batch[edge_index[0]]
                per_type = torch.zeros(bsz, device=device)
                cnt = torch.zeros(bsz, device=device)
                per_type.scatter_add_(0, edge_batch, per_edge)
                cnt.scatter_add_(0, edge_batch, torch.ones_like(per_edge))
                has = cnt > 0
                per_graph[has] += (per_type[has] / cnt[has]) * w
                edge_w_sum[has] += w
            per_graph = per_graph / edge_w_sum.clamp(min=1e-8)
            all_losses.extend(per_graph.detach().cpu().tolist())
    model.train(was_training)
    losses_t = torch.tensor(all_losses, dtype=torch.float64)
    total = losses_t.sum()
    if total <= 0 or not torch.isfinite(total):
        return torch.full((len(dataset),), 1.0 / max(1, len(dataset)), dtype=torch.float64)
    return losses_t / total


def build_stratified_weights(snsb_train):
    from collections import Counter
    global_counts = Counter()
    per_graph_labels = []
    for i in range(len(snsb_train)):
        sample = snsb_train[i]
        labels = sample["bystander"].y.long().tolist()
        per_graph_labels.append(labels)
        for c in labels:
            global_counts[c] += 1
    weights = []
    for labels in per_graph_labels:
        if not labels:
            weights.append(1.0)
        else:
            rarest = min(labels, key=lambda c: global_counts.get(c, 1))
            weights.append(1.0 / (global_counts[rarest] ** 0.5))
    wt = torch.tensor(weights, dtype=torch.float64)
    print(f"  Stratified weights: {len(weights)} graphs, class counts={dict(global_counts)}")
    return wt / wt.sum()


def next_cycled(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    persistent = num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers, persistent_workers=persistent,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# Main train
# ---------------------------------------------------------------------------

def train(args):
    seed_label = "none" if args.seed is None else str(args.seed)
    if wandb is not None and args.wandb:
        wandb.init(project=args.wandb_project,
                   name=args.run_name or f"unified_v3_s{seed_label}",
                   config=vars(args))
    elif args.wandb:
        print("wandb not installed; continuing without W&B.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}  seed={seed_label}")

    # ---- SEAN: combined_v1.h5 + split_indices_final.pt ----
    combined_ds = UnifiedGraphDataset(args.h5_path, ssub=False)
    split = torch.load(args.split_file, map_location="cpu", weights_only=False)

    sean_train = [i for i in split["train"] if combined_ds._dataset_ids[i] == SEAN_ID]
    sean_val   = [i for i in split["val"]   if combined_ds._dataset_ids[i] == SEAN_ID]
    sean_test  = [i for i in split["test"]  if combined_ds._dataset_ids[i] == SEAN_ID]

    combined_ds.set_robot_stats(sean_train)
    print(f"SEAN  train={len(sean_train)}  val={len(sean_val)}  test={len(sean_test)}")

    # ---- SNSB: social_nav_v3.h5 at native T=20, scene-matched to the fixed split ----
    print(f"Loading SNSB from {args.snsb_h5_path} ...")
    snsb_ds = UnifiedSocialNavDataset(args.snsb_h5_path, zero_node_features=True)
    snsb_train_idx, snsb_val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)
    # no set_feature_stats: SNSB node features not used — model uses edge-based init

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    split_path = os.path.join(args.checkpoint_dir, f"split_used_v3_seed{seed_label}.pt")
    torch.save({
        "source_combined": args.h5_path,
        "source_snsb": args.snsb_h5_path,
        "split_file": args.split_file,
        "sean_train": sean_train, "sean_val": sean_val, "sean_test": sean_test,
        "snsb_train": snsb_train_idx, "snsb_val": snsb_val_idx,
    }, split_path)
    print(f"Split saved → {split_path}")

    # ---- Subsets + loaders ----
    sean_train_wrap = SEANHeteroWrapper(torch.utils.data.Subset(combined_ds, sean_train))
    sean_val_wrap   = SEANHeteroWrapper(torch.utils.data.Subset(combined_ds, sean_val))
    snsb_train_sub  = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_ds, snsb_train_idx))
    snsb_val_sub    = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_ds, snsb_val_idx))

    snsb_sampler = None
    sean_sampler = None
    stratified_weights = None
    sean_uniform_weights = None
    if args.weighted_sampling:
        print("Computing stratified sampling weights (SNSB)...")
        stratified_weights = build_stratified_weights(snsb_train_sub)
        snsb_sampler = WeightedRandomSampler(
            stratified_weights, num_samples=len(snsb_train_sub), replacement=True
        )
        sean_uniform_weights = torch.full((len(sean_train_wrap),), 1.0 / len(sean_train_wrap), dtype=torch.float64)
        sean_sampler = WeightedRandomSampler(
            sean_uniform_weights, num_samples=len(sean_train_wrap), replacement=True
        )
        print(f"Weighted sampling: SNSB train={len(snsb_train_sub)}, SEAN train={len(sean_train_wrap)}")

    sean_train_loader = make_loader(sean_train_wrap, args.batch_size, True,  args.num_workers, sampler=sean_sampler)
    sean_val_loader   = make_loader(sean_val_wrap,   args.batch_size, False, args.num_workers)
    snsb_train_loader = make_loader(snsb_train_sub,  args.batch_size, True,  args.num_workers, sampler=snsb_sampler)
    snsb_val_loader   = make_loader(snsb_val_sub,    args.batch_size, False, args.num_workers)

    # ---- Model: snsb_timesteps=20, no SNSB 60D node features, edge-initialized type nodes ----
    model = UnifiedGNNVAE(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    es_counter = 0
    best_path = os.path.join(args.checkpoint_dir, f"best_unified_gnn_vae_v3_seed{seed_label}.pt")
    step_count = args.epoch_steps or max(len(sean_train_loader), len(snsb_train_loader))
    loss_weight_sum = args.sean_loss_weight + args.snsb_loss_weight

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / max(1, args.warmup_epochs))

        if args.weighted_sampling and epoch > args.adaptive_warmup_epochs:
            epochs_since = epoch - args.adaptive_warmup_epochs
            alpha = min(args.adaptive_max_alpha,
                        args.adaptive_max_alpha * epochs_since / max(args.adaptive_ramp_epochs, 1))
            snsb_loss_w = compute_per_sample_losses(
                model, snsb_train_sub, device, args.batch_size, args.num_workers, "snsb"
            )
            snsb_combined = (1.0 - alpha) * stratified_weights + alpha * snsb_loss_w
            snsb_sampler.weights = (snsb_combined / snsb_combined.sum()).double()
            sean_loss_w = compute_per_sample_losses(
                model, sean_train_wrap, device, args.batch_size, args.num_workers, "sean"
            )
            sean_combined = (1.0 - alpha) * sean_uniform_weights + alpha * sean_loss_w
            sean_sampler.weights = (sean_combined / sean_combined.sum()).double()

        model.train()
        sean_iter = iter(sean_train_loader)
        snsb_iter = iter(snsb_train_loader)
        sums = {k: 0.0 for k in [
            "sean_total", "sean_r_robot", "sean_r_edge", "sean_kl",
            "snsb_total", "snsb_r_robot", "snsb_r_edge", "snsb_kl",
        ]}

        for _ in range(step_count):
            sean_batch, sean_iter = next_cycled(sean_iter, sean_train_loader)
            snsb_batch, snsb_iter = next_cycled(snsb_iter, snsb_train_loader)
            sean_batch = sean_batch.to(device)
            snsb_batch = snsb_batch.to(device)

            optimizer.zero_grad()

            sean_recon, sean_edges, sean_mu, sean_lv = model(sean_batch)
            sean_loss, sean_rr, sean_re, sean_kl = compute_loss(
                sean_recon, sean_edges, sean_batch, sean_mu, sean_lv, beta
            )

            snsb_recon, snsb_edges, snsb_mu, snsb_lv = model(snsb_batch)
            snsb_loss, snsb_rr, snsb_re, snsb_kl = compute_loss(
                snsb_recon, snsb_edges, snsb_batch, snsb_mu, snsb_lv, beta
            )

            loss = (args.sean_loss_weight * sean_loss + args.snsb_loss_weight * snsb_loss) / loss_weight_sum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            sums["sean_total"] += float(sean_loss); sums["sean_r_robot"] += float(sean_rr)
            sums["sean_r_edge"] += float(sean_re);  sums["sean_kl"]      += float(sean_kl)
            sums["snsb_total"] += float(snsb_loss); sums["snsb_r_robot"] += float(snsb_rr)
            sums["snsb_r_edge"] += float(snsb_re);  sums["snsb_kl"]      += float(snsb_kl)

        tm = {k: v / max(1, step_count) for k, v in sums.items()}
        sean_val_m = run_eval(model, sean_val_loader, device, beta)
        snsb_val_m = run_eval(model, snsb_val_loader, device, beta)
        val_total = (args.sean_loss_weight * sean_val_m["total"]
                     + args.snsb_loss_weight * snsb_val_m["total"]) / loss_weight_sum
        scheduler.step(val_total)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta={beta:.4f} | "
            f"sean={tm['sean_total']:.4f}(r={tm['sean_r_robot']:.3f},e={tm['sean_r_edge']:.3f},kl={tm['sean_kl']:.3f}) | "
            f"snsb={tm['snsb_total']:.4f}(r={tm['snsb_r_robot']:.3f},e={tm['snsb_r_edge']:.3f},kl={tm['snsb_kl']:.3f}) | "
            f"val sean={sean_val_m['total']:.4f} snsb={snsb_val_m['total']:.4f} combined={val_total:.4f}"
        )

        if wandb is not None and args.wandb:
            wandb.log({
                "epoch": epoch, "beta": beta, "lr": optimizer.param_groups[0]["lr"],
                "train/sean_total": tm["sean_total"], "train/sean_r_robot": tm["sean_r_robot"],
                "train/sean_r_edge": tm["sean_r_edge"], "train/sean_kl": tm["sean_kl"],
                "train/snsb_total": tm["snsb_total"], "train/snsb_r_robot": tm["snsb_r_robot"],
                "train/snsb_r_edge": tm["snsb_r_edge"], "train/snsb_kl": tm["snsb_kl"],
                "val/sean_total": sean_val_m["total"], "val/snsb_total": snsb_val_m["total"],
                "val/combined": val_total,
            }, step=epoch)

        if val_total < best_val_loss:
            best_val_loss = val_total
            es_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_total": val_total,
                "val_sean": sean_val_m, "val_snsb": snsb_val_m,
                "split_path": split_path,
                "h5_path": args.h5_path,
                "snsb_h5_path": args.snsb_h5_path,
                "sean_robot_mean": combined_ds.robot_mean,
                "sean_robot_std":  combined_ds.robot_std,
                "snsb_timesteps": 20,
                "zero_node_features": True,
                "snsb_edge_node_init": True,
                "args": vars(args),
            }, best_path)
            print(f"  Saved best → {best_path}")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.es_patience})")
                break

    if wandb is not None and args.wandb:
        wandb.finish()
    print(f"Done. Best combined val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",    default="/data/combined_v1.h5",
                        help="SEAN source (combined_v1.h5)")
    parser.add_argument("--snsb_h5_path", default="/data/social_nav_v3.h5",
                        help="SNSB source at native T=20 (social_nav_v3.h5)")
    parser.add_argument("--split_file", default="checkpoints/split_indices_final.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--seed", type=lambda s: None if s.lower() == "none" else int(s), default=0,
                        help="Torch RNG seed. Use '--seed none' to leave model/loader RNG unseeded.")
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--es_patience", type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--latent_dim",  type=int,   default=64)
    parser.add_argument("--num_mp_layers", type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--beta_max",    type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",   type=float, default=20.0)
    parser.add_argument("--epoch_steps", type=int,   default=None)
    parser.add_argument("--sean_loss_weight", type=float, default=1.0)
    parser.add_argument("--snsb_loss_weight", type=float, default=1.0)
    parser.add_argument("--weighted_sampling", action="store_true")
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=20)
    parser.add_argument("--adaptive_max_alpha",     type=float, default=0.5)
    parser.add_argument("--adaptive_ramp_epochs",   type=int, default=50)
    parser.add_argument("--cpu",   action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="social_world_model")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    train(args)
