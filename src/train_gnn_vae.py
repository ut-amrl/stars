"""
Training script for GNN-VAE on SEAN_TOGETHER social scene data.

Loss:
  total = recon_loss + beta * kl_loss
  recon_loss = MSE between reconstructed and original node features
  kl_loss    = KL divergence of q(z|x) from N(0,1)

Beta annealing: beta starts at 0 and ramps up to beta_max over
  warmup_epochs so the model first learns to reconstruct before
  being penalised for the latent structure.

Usage:
  python train_gnn_vae.py
  python train_gnn_vae.py --epochs 50 --latent_dim 64 --beta_max 1.0
  python train_gnn_vae.py --h5_path dataloaders/sean_together.h5 --checkpoint_dir checkpoints
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.sean_graph_dataset import SEANGraphDataset
from models.gnn_vae import GNNVAE


def vae_loss(x_recon, x, mu, logvar, beta):
    recon = F.mse_loss(x_recon, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Data ----
    print(f"Loading dataset from {args.h5_path}...")
    dataset = SEANGraphDataset(args.h5_path)

    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    print(f"Train: {train_size}  |  Val: {val_size}")

    # ---- Model ----
    model = GNNVAE(
        node_dim=5,
        edge_dim=3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        # Beta annealing: ramp from 0 -> beta_max over warmup_epochs
        beta = min(args.beta_max, args.beta_max * epoch / max(1, args.warmup_epochs))

        # ---- Train ----
        model.train()
        train_recon = train_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_recon, mu, logvar, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            loss, recon, kl = vae_loss(x_recon, batch.x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            train_recon += recon.item()
            train_kl    += kl.item()

        train_recon /= len(train_loader)
        train_kl    /= len(train_loader)

        # ---- Val ----
        model.eval()
        val_recon = val_kl = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_recon, mu, logvar, _ = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                _, recon, kl = vae_loss(x_recon, batch.x, mu, logvar, beta)
                val_recon += recon.item()
                val_kl    += kl.item()

        val_recon /= len(val_loader)
        val_kl    /= len(val_loader)
        val_total  = val_recon + beta * val_kl

        scheduler.step(val_total)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta={beta:.3f} | "
            f"train recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val recon={val_recon:.4f} kl={val_kl:.4f}"
        )

        # ---- Checkpoint ----
        if val_total < best_val_loss:
            best_val_loss = val_total
            path = os.path.join(args.checkpoint_dir, "best_gnn_vae.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_recon": val_recon,
                "val_kl": val_kl,
                "args": vars(args),
            }, path)
            print(f"  Saved best checkpoint (val_loss={val_total:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/best_gnn_vae.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",        default="dataloaders/sean_together.h5")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--latent_dim",     type=int,   default=64)
    parser.add_argument("--num_mp_layers",  type=int,   default=3)
    parser.add_argument("--beta_max",       type=float, default=1.0)
    parser.add_argument("--warmup_epochs",  type=int,   default=10,
                        help="Epochs to ramp beta from 0 to beta_max")
    args = parser.parse_args()
    train(args)
