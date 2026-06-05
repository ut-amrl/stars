"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 VAE trainer.

Training script for GNN-VAE v4 - Stage 1: fully self-supervised VAE.

Changes from v3:
  - Uses SEANGraphDatasetV4: follower has no GT features (num_nodes=1 only)
  - Uses GNNVAEv4: follower initialized from learnable 8D embed, like bystander
  - Loss = recon_robot + recon_edges + beta*KL  (no recon_follower)
  - forward() returns (robot_recon, edge_recons, mu_dict, logvar_dict)

Loss = recon_robot + recon_edges + beta*KL

Usage:
  cd src/
  python train_gnn_vae_v4.py
  python train_gnn_vae_v4.py --epochs 100 --batch_size 32
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.sean_graph_dataset_v4 import SEANGraphDatasetV4, EDGE_TYPES
from models.gnn_vae_v4 import GNNVAEv4, NODE_TYPES

# Edge reconstruction loss weights - downweight b->b (70% of edges, background info)
EDGE_LOSS_WEIGHTS = {
    ('robot',     'to', 'follower'):  5.0,
    ('follower',  'to', 'robot'):     5.0,
    ('robot',     'to', 'bystander'): 2.0,
    ('bystander', 'to', 'robot'):     2.0,
    ('follower',  'to', 'bystander'): 1.0,
    ('bystander', 'to', 'follower'):  1.0,
    ('bystander', 'to', 'bystander'): 0.1,
}


def get_checkpoint_filename(args):
    """Stage 1 checkpoint filename for the current aux-pooling configuration."""
    tags = []
    if args.incl_pool_attn:
        tags.append("pool_attn")
    if args.incl_mlp_pool:
        tags.append("mlp_pool")
    if not tags:
        return "best_gnn_vae_v4.pt"
    return f"best_gnn_vae_v4_{'_'.join(tags)}.pt"


def compute_loss(robot_recon, edge_recons, data, mu_dict, logvar_dict, beta):
    # Robot node recon
    recon_robot = F.mse_loss(robot_recon, data['robot'].x)

    # Edge recon - weighted MSE across edge types (downweight b->b)
    edge_losses  = []
    edge_weights = []
    for et in EDGE_TYPES:
        if et in edge_recons and hasattr(data[et], 'edge_attr'):
            edge_losses.append(F.mse_loss(edge_recons[et], data[et].edge_attr))
            edge_weights.append(EDGE_LOSS_WEIGHTS[et])
    if edge_losses:
        w           = torch.tensor(edge_weights, device=data['robot'].x.device)
        losses      = torch.stack(edge_losses)
        recon_edges = (w * losses).sum() / w.sum()
    else:
        recon_edges = torch.tensor(0.0, device=data['robot'].x.device)

    # KL - over all per-node z's concatenated
    all_mu     = torch.cat([mu_dict[t] for t in NODE_TYPES if mu_dict[t].shape[0] > 0], dim=0)
    all_logvar = torch.cat([logvar_dict[t] for t in NODE_TYPES if logvar_dict[t].shape[0] > 0], dim=0)
    kl = -0.5 * torch.mean(1 + all_logvar - all_mu.pow(2) - all_logvar.exp())

    recon = recon_robot + recon_edges
    total = recon + beta * kl
    return total, recon_robot, recon_edges, kl


def train(args):
    ckpt_name = get_checkpoint_filename(args)
    if wandb is not None and args.wandb:
        wandb.init(project="social_world_model", name=f"gnn_vae_v4_e{args.epochs}", config=vars(args))
    else:
        print("wandb not active; continuing without experiment logging.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading dataset from {args.h5_path}...")
    dataset = SEANGraphDatasetV4(args.h5_path)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.split_file is not None:
        # Load pre-computed split (e.g. scene-level split from generate_sean_snsb_hetero_split.py)
        saved = torch.load(args.split_file, map_location="cpu")
        train_indices = list(saved["train"])
        val_indices   = list(saved["val"])
        test_indices  = list(saved["test"])
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set   = torch.utils.data.Subset(dataset, val_indices)
        test_set  = torch.utils.data.Subset(dataset, test_indices)
        train_size, val_size, test_size = len(train_indices), len(val_indices), len(test_indices)
        print(f"Loaded split from {args.split_file}")
    else:
        N          = len(dataset)
        test_size  = int(0.04 * N)
        val_size   = int(0.16 * N)
        train_size = N - val_size - test_size
        generator = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    # Compute normalization stats from train indices only (no leakage from val/test)
    dataset.set_robot_stats(train_set.indices)

    # Save split indices for Stage 2 linear probe
    split_path = os.path.join(args.checkpoint_dir, "split_indices_v4.pt")
    torch.save({
        'train': list(train_set.indices),
        'val':   list(val_set.indices),
        'test':  list(test_set.indices),
    }, split_path)
    print(f"Split indices saved -> {split_path}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=persistent)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=persistent)
    print(f"Train: {train_size}  |  Val: {val_size}  |  Test: {test_size}")

    model = GNNVAEv4(
        robot_dim=592,
        follower_embed_dim=8,
        bystander_embed_dim=8,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
        edge_transformer_layers=args.edge_transformer_layers,
        edge_transformer_heads=args.edge_transformer_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Aux task heads: predict mean robot->follower distance from pooled z output
    aux_head     = None
    mlp_aux_head = None
    if args.incl_pool_attn:
        aux_head = torch.nn.Linear(3 * args.latent_dim, 1).to(device)
        print(f"pool_attn aux task enabled (aux_decay={args.aux_decay})")
    if args.incl_mlp_pool:
        mlp_aux_head = torch.nn.Linear(3 * args.latent_dim, 1).to(device)
        print(f"mlp_pool aux task enabled (aux_decay={args.aux_decay})")

    params    = list(model.parameters()) + \
                (list(aux_head.parameters()) if aux_head else []) + \
                (list(mlp_aux_head.parameters()) if mlp_aux_head else [])
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    es_counter    = 0

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / max(1, args.warmup_epochs))

        # Training VAE model and Aux tasks (MLP or attn_pool of Node z's if needed for linear probe)
        # Currently: Aux task is to minimize dist between pooled Z and mean distance between robot and follower
        model.train()
        if aux_head:     aux_head.train()
        if mlp_aux_head: mlp_aux_head.train()
        t_r_rob = t_r_edge = t_kl = t_aux = t_mlp_aux = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
            loss, r_rob, r_edge, kl = compute_loss(
                robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta,
            )
            rf_attr = batch[('robot', 'to', 'follower')].edge_attr              # [B, 40, 8]
            rf_dist = (rf_attr[:,:,0]**2 + rf_attr[:,:,1]**2).sqrt().mean(1, keepdim=True)  # [B,1]
            if aux_head is not None:
                z_graph = model.pool_z(mu_dict, batch)                              # [B, 3*latent_dim]
                aux_loss = F.mse_loss(aux_head(z_graph), rf_dist)
                loss = loss + args.aux_decay * aux_loss
                t_aux += aux_loss.item()
            if mlp_aux_head is not None:
                z_mlp    = model.pool_z_mlp(mu_dict, batch)                         # [B, 3*latent_dim]
                mlp_loss = F.mse_loss(mlp_aux_head(z_mlp), rf_dist)
                loss = loss + args.aux_decay * mlp_loss
                t_mlp_aux += mlp_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            t_r_rob  += r_rob.item()
            t_r_edge += r_edge.item()
            t_kl     += kl.item()

        n = len(train_loader)
        t_r_rob /= n; t_r_edge /= n; t_kl /= n; t_aux /= n; t_mlp_aux /= n

        # ---- Val ----
        model.eval()
        if aux_head:     aux_head.eval()
        if mlp_aux_head: mlp_aux_head.eval()
        v_r_rob = v_r_edge = v_kl = v_aux = v_mlp_aux = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
                _, r_rob, r_edge, kl = compute_loss(
                    robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta,
                )
                rf_attr = batch[('robot', 'to', 'follower')].edge_attr
                rf_dist = (rf_attr[:,:,0]**2 + rf_attr[:,:,1]**2).sqrt().mean(1, keepdim=True)
                if aux_head is not None:
                    z_graph = model.pool_z(mu_dict, batch)
                    v_aux += F.mse_loss(aux_head(z_graph), rf_dist).item()
                if mlp_aux_head is not None:
                    z_mlp    = model.pool_z_mlp(mu_dict, batch)
                    v_mlp_aux += F.mse_loss(mlp_aux_head(z_mlp), rf_dist).item()
                v_r_rob  += r_rob.item()
                v_r_edge += r_edge.item()
                v_kl     += kl.item()

        n = len(val_loader)
        v_r_rob /= n; v_r_edge /= n; v_kl /= n; v_aux /= n; v_mlp_aux /= n
        val_total = v_r_rob + v_r_edge + beta * v_kl

        scheduler.step(val_total)

        aux_str     = f" aux={t_aux:.4f}/{v_aux:.4f}" if aux_head else ""
        mlp_aux_str = f" mlp_aux={t_mlp_aux:.4f}/{v_mlp_aux:.4f}" if mlp_aux_head else ""
        aux_str     = aux_str + mlp_aux_str
        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta={beta:.4f} | "
            f"train  r_robot={t_r_rob:.3f} r_edge={t_r_edge:.3f} kl={t_kl:.3f} | "
            f"val    r_robot={v_r_rob:.3f} r_edge={v_r_edge:.3f} kl={v_kl:.3f}{aux_str}"
        )

        log_dict = {
            "beta":          beta,
            "train/r_robot": t_r_rob,
            "train/r_edge":  t_r_edge,
            "train/kl":      t_kl,
            "train/total":   t_r_rob + t_r_edge + beta * t_kl,
            "val/r_robot":   v_r_rob,
            "val/r_edge":    v_r_edge,
            "val/kl":        v_kl,
            "val/total":     val_total,
            "lr":            optimizer.param_groups[0]['lr'],
        }
        if aux_head:
            log_dict["train/aux"]     = t_aux
            log_dict["val/aux"]       = v_aux
        if mlp_aux_head:
            log_dict["train/mlp_aux"] = t_mlp_aux
            log_dict["val/mlp_aux"]   = v_mlp_aux
        if wandb is not None and args.wandb:
            wandb.log(log_dict, step=epoch)

        if val_total < best_val_loss:
            best_val_loss = val_total
            es_counter    = 0
            path = os.path.join(args.checkpoint_dir, ckpt_name)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_total":        val_total,
                "val_r_robot":      v_r_rob,
                "val_r_edges":      v_r_edge,
                "val_kl":           v_kl,
                "robot_mean":       dataset.robot_mean,
                "robot_std":        dataset.robot_std,
                "split_path":       split_path,
                "args":             vars(args),
            }, path)
            print(f"  Saved best checkpoint (val_loss={val_total:.4f})")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.es_patience})")
                break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {args.checkpoint_dir}/{ckpt_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",        default="/data/sean_together_v2.h5")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--split_file",     default=None,
                        help="Path to a .pt file with pre-computed train/val/test index lists "
                             "(e.g. checkpoints/split_indices_sean_snsb_hetero_v1.pt). "
                             "If omitted, uses 80/16/4 random_split with seed=42.")
    parser.add_argument("--epochs",         type=int,   default=1000)
    parser.add_argument("--es_patience",    type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--latent_dim",     type=int,   default=64)
    parser.add_argument("--num_mp_layers",  type=int,   default=2)
    parser.add_argument("--edge_transformer_layers", type=int, default=2)
    parser.add_argument("--edge_transformer_heads",  type=int, default=2)
    parser.add_argument("--beta_max",       type=float, default=0.1)
    parser.add_argument("--warmup_epochs",  type=int,   default=50)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--grad_clip",      type=float, default=20.0)
    parser.add_argument("--incl_pool_attn", action="store_true",
                        help="Enable aux task: train pool_attn via 192D pool_z -> predict rf distance")
    parser.add_argument("--incl_mlp_pool",  action="store_true",
                        help="Enable aux task: train pool_mlp (2-layer MLP on 192D concat) -> predict rf distance")
    parser.add_argument("--aux_decay",      type=float, default=0.1,
                        help="Weight on aux task loss (used by --incl_pool_attn and/or --incl_mlp_pool)")
    parser.add_argument("--wandb",          action="store_true",
                        help="Enable Weights & Biases logging")
    args = parser.parse_args()
    train(args)
