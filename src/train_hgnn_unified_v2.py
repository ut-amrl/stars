import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

try:
    import wandb
except ImportError:
    wandb = None

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_all import UnifiedGraphDataset
from models.hgnn_combined_v2 import HGNNCombinedVAE as Model

IGNORE_LABEL       = -100
NUM_ACTION_CLASSES = 5


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg["homognn_params"])


def create_run_dir(experiments_root, run_name):
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name      = str(run_name).strip()
    run_dir   = os.path.join(experiments_root, f"{timestamp}_{name}" if name else timestamp)
    suffix = 1
    base   = run_dir
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"
    os.makedirs(run_dir)
    return run_dir


def resolve_device(device_cfg):
    s = str(device_cfg).strip().lower()

    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if s == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Requested CUDA but no CUDA device is available; falling back to CPU.")
        return torch.device("cpu")

    if s.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(
                f"Requested {s} but no CUDA device is available; falling back to CPU."
            )
            return torch.device("cpu")

        try:
            requested_idx = int(s.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device_cfg}") from exc

        visible_count = torch.cuda.device_count()
        if requested_idx < 0 or requested_idx >= visible_count:
            print(
                f"Requested {s}, but only {visible_count} CUDA device(s) are visible; "
                "falling back to cuda:0."
            )
            return torch.device("cuda:0")

        return torch.device(s)

    return torch.device(s)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_stratified_weights(train_subset):
    n            = len(train_subset)
    class_counts = torch.zeros(NUM_ACTION_CLASSES, dtype=torch.float64)
    per_graph    = []

    for i in range(n):
        data         = train_subset[i]
        human_labels = data.action_label[data.action_label != IGNORE_LABEL]
        per_graph.append(human_labels)
        for c in range(NUM_ACTION_CLASSES):
            class_counts[c] += (human_labels == c).sum().item()

    total    = class_counts.sum()
    inv_freq = (total / class_counts.clamp(min=1)).sqrt()

    weights = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        labels = per_graph[i]
        if labels.numel() == 0:
            weights[i] = 1.0
        else:
            weights[i] = inv_freq[labels.unique()].max().item()

    return (weights / weights.sum()).float()


def compute_per_sample_losses(model, dataset, device, batch_size, ssub):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_losses = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            edge_recon, node_feat_recon, mu, logvar = model(batch)
            if ssub:
                per_edge = ((edge_recon - batch.edge_attr) ** 2).mean(dim=(-2, -1))
            else:
                src, dst   = batch.edge_index[0], batch.edge_index[1]
                edge_valid = batch.valid_mask[src] * batch.valid_mask[dst]
                per_edge   = ((edge_recon - batch.edge_attr) ** 2
                              * edge_valid.unsqueeze(-1)).mean(dim=(-2, -1))

            edge_batch     = batch.batch[batch.edge_index[0]]
            B              = int(batch.batch.max()) + 1
            per_graph_loss = scatter(per_edge, edge_batch, dim=0,
                                     reduce="mean", dim_size=B)
            all_losses.append(per_graph_loss.cpu())

    return torch.cat(all_losses, dim=0)


def run_epoch(model, loader, device, ssub, node_loss_weight, beta,
              optimizer=None, grad_clip=None, use_amp=False, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss      = 0.0
    total_edge_loss = 0.0
    total_node_loss = 0.0
    total_kl_loss   = 0.0
    n_batches       = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                edge_recon, node_feat_recon, mu, logvar = model(batch)

                if ssub:
                    loss_edge  = F.mse_loss(edge_recon, batch.edge_attr)
                    loss_node  = torch.tensor(0.0, device=device)
                else:
                    src, dst   = batch.edge_index[0], batch.edge_index[1]
                    edge_valid = batch.valid_mask[src] * batch.valid_mask[dst]
                    loss_edge  = (
                        (edge_recon - batch.edge_attr) ** 2 * edge_valid.unsqueeze(-1)
                    ).sum() / edge_valid.unsqueeze(-1).sum().clamp(min=1.0)

                    feat_mask = batch.x.abs().sum(dim=-1) > 0
                    if feat_mask.any():
                        loss_node = F.mse_loss(node_feat_recon[feat_mask], batch.x[feat_mask])
                    else:
                        loss_node = torch.tensor(0.0, device=device)

                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_edge + node_loss_weight * loss_node + beta * loss_kl

            if training:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            print(f"CUDA OOM on batch {batch_idx}; skipping batch and clearing cache.")
            if training:
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        total_loss      += float(loss.detach().cpu())
        total_edge_loss += float(loss_edge.detach().cpu())
        total_node_loss += float(loss_node.detach().cpu())
        total_kl_loss   += float(loss_kl.detach().cpu())
        n_batches       += 1

    if n_batches == 0:
        raise RuntimeError(
            "All batches in this epoch were skipped due to CUDA OOM. "
            "Reduce batch_size or model size."
        )

    n = max(1, n_batches)
    return total_loss / n, total_edge_loss / n, total_node_loss / n, total_kl_loss / n


def train(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}  |  Mode: {'SSUB-only' if args.ssub else 'Combined'}")

    requested_amp = bool(getattr(args, "use_amp", False))
    use_amp       = requested_amp and device.type == "cuda"
    if requested_amp and not use_amp:
        print("AMP requested but CUDA is not in use; AMP disabled.")
    if use_amp:
        print("AMP enabled (torch.cuda.amp.autocast)")

    dataset = UnifiedGraphDataset(args.h5_path, ssub=args.ssub)

    if getattr(args, "split_file", None) is not None:
        saved = torch.load(args.split_file, map_location="cpu")
        train_idx = list(saved["train"])
        val_idx   = list(saved["val"])
        test_idx  = list(saved["test"])
        print(f"Loaded split from {args.split_file}")
    else:
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        train_idx, val_idx, test_idx = dataset.scene_split(
            val_ratio=args.val_ratio, test_ratio=test_ratio, seed=args.seed,
        )
    dataset.source_counts(train_idx, val_idx, test_idx)

    if not args.ssub:
        dataset.set_robot_stats(train_idx)

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)

    split_path = os.path.join(args.run_dir, "split_indices.pt")
    torch.save({"train": train_idx, "val": val_idx, "test": test_idx}, split_path)
    print(f"Split saved → {split_path}")

    use_weighted  = args.use_weighted_sampling
    use_adaptive  = args.adaptive_reweight
    warmup_epochs = args.adaptive_warmup_epochs
    max_alpha     = args.adaptive_max_alpha

    sampler            = None
    stratified_weights = None
    if use_weighted:
        print("Computing stratified sampling weights...")
        stratified_weights = compute_stratified_weights(train_set)
        print(
            f"  Weight range: [{stratified_weights.min():.6f}, "
            f"{stratified_weights.max():.6f}]  "
            f"std={stratified_weights.std():.6f}"
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=stratified_weights,
            num_samples=len(train_set),
            replacement=True,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    model = Model(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
        edge_transformer_layers=args.edge_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── W&B
    use_wandb = bool(args.wandb)
    if use_wandb and wandb is None:
        print("wandb not installed — continuing without W&B.")
        use_wandb = False
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or os.path.basename(args.run_dir),
            config=vars(args),
            dir=args.run_dir,
        )
    else:
        print("wandb logging disabled.")

    best_val_loss     = float("inf")
    best_path         = os.path.join(args.run_dir, "best_model.pt")
    patience          = args.early_stopping_patience
    min_delta         = args.early_stopping_min_delta
    epochs_no_improve = 0
    node_loss_weight  = args.node_loss_weight

    beta_max = getattr(args, "beta_max", 0.1)
    beta_warmup = getattr(args, "beta_warmup_epochs", getattr(args, "warmup_epochs", 50))

    for epoch in range(1, args.epochs + 1):
        beta = min(beta_max, beta_max * epoch / max(1, beta_warmup))

        if (use_weighted and use_adaptive and sampler is not None
                and stratified_weights is not None
                and epoch > warmup_epochs):
            loss_weights = compute_per_sample_losses(
                model, train_set, device, args.batch_size, args.ssub
            )
            loss_weights = loss_weights / loss_weights.sum()
            epochs_since = epoch - warmup_epochs
            ramp_epochs  = args.adaptive_ramp_epochs
            alpha        = min(max_alpha, max_alpha * epochs_since / max(ramp_epochs, 1))
            combined = (1.0 - alpha) * stratified_weights + alpha * loss_weights
            combined = combined / combined.sum()
            sampler.weights = combined.double()
            ess = float(1.0 / (combined ** 2).sum())
            print(f"  Adaptive reweight: alpha={alpha:.3f}  ESS={ess:.0f}/{len(train_set)}")

        train_loss, train_edge, train_node, train_kl = run_epoch(
            model, train_loader, device, args.ssub, node_loss_weight, beta,
            optimizer=optimizer, grad_clip=args.grad_clip,
            use_amp=use_amp, scaler=scaler,
        )
        with torch.no_grad():
            val_loss, val_edge, val_node, val_kl = run_epoch(
                model, val_loader, device, args.ssub, node_loss_weight, beta,
                use_amp=use_amp,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta={beta:.4f} | "
            f"train={train_loss:.6f} (edge={train_edge:.6f} node={train_node:.6f} kl={train_kl:.6f}) | "
            f"val={val_loss:.6f} (edge={val_edge:.6f} node={val_node:.6f} kl={val_kl:.6f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        log_dict = {
            "epoch": epoch,
            "beta": beta,
            "train/loss": train_loss, "train/loss_edge": train_edge,
            "train/loss_node": train_node, "train/kl": train_kl,
            "val/loss":   val_loss,   "val/loss_edge":   val_edge,
            "val/loss_node":   val_node, "val/kl": val_kl,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if use_weighted and use_adaptive and epoch > warmup_epochs:
            log_dict["sampling/alpha"] = alpha
            log_dict["sampling/ess"]   = ess
        if use_wandb:
            wandb.log(log_dict, step=epoch)

        if val_loss < best_val_loss - min_delta:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_dict = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "train_loss":       train_loss,
                "split_path":       split_path,
                "ssub":             args.ssub,
                "args":             vars(args),
            }
            if not args.ssub:
                save_dict["robot_mean"] = dataset.robot_mean
                save_dict["robot_std"]  = dataset.robot_std
            torch.save(save_dict, best_path)
            print(f"  Saved best → {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if use_wandb:
        wandb.finish()

    print(f"Done. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params/config.yaml")
    parser.add_argument("--ssub", action="store_true",
                        help="SSUB-only mode (no valid_mask, no node features)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--num_mp_layers", type=int, default=None)
    parser.add_argument("--edge_transformer_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--seed", type=lambda s: None if s.lower() == "none" else int(s), default=None)
    parser.add_argument("--h5_path", default=None)
    parser.add_argument("--experiments_root", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--beta_warmup_epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--weighted_sampling", action="store_true", default=None)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=None)
    parser.add_argument("--adaptive_max_alpha", type=float, default=None)
    parser.add_argument("--adaptive_ramp_epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    cli_args = parser.parse_args()

    args             = load_config(cli_args.config)
    args.ssub        = cli_args.ssub
    args.config_path = os.path.abspath(cli_args.config)
    
    # Overwrite loaded config parameters with command-line ones if they were provided
    for key, val in vars(cli_args).items():
        if val is not None and key != "config":
            if key == "weighted_sampling":
                args.use_weighted_sampling = val
            else:
                setattr(args, key, val)

    args.h5_path     = os.path.abspath(args.h5_path)
    args.experiments_root = os.path.abspath(args.experiments_root)
    
    # Handle checkpoint directory
    if cli_args.checkpoint_dir is not None:
        args.run_dir = os.path.abspath(cli_args.checkpoint_dir)
        os.makedirs(args.run_dir, exist_ok=True)
    else:
        args.run_dir = create_run_dir(args.experiments_root, args.run_name)

    shutil.copy2(args.config_path, os.path.join(args.run_dir, "config.yaml"))
    with open(os.path.join(args.run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    train(args)
