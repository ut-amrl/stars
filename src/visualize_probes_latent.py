"""
Visualization of Latent Spaces via t-SNE and PCA.

Extracts test-split latent representations from three downstream tasks:
2. SNS (human node-level VAE embeddings, 128D)

Performs dimensionality reduction to 2D using:
- t-SNE (non-linear)
- PCA (linear)

Generates individual categorical plots and unified combined plots for each dataset
with premium, state-of-the-art visual design.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from types import SimpleNamespace

# Setup paths and environment
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

# Add group_gnn to path
_possible_group_gnn_dirs = [
    os.path.abspath(os.path.join(SRC_DIR, "..", "..", "group_gnn")),
    os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", "group_gnn")),
    "/home/ntsoi/all_swm/group_gnn"
]
GROUP_GNN_DIR = None
for _d in _possible_group_gnn_dirs:
    if os.path.exists(_d):
        GROUP_GNN_DIR = _d
        sys.path.append(_d)
        break

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from probes.common import load_frozen_encoder, resolve_device, seed_everything
from probes.sns_probe import ACTION_CLASSES, IGNORE_LABEL
    extract_vae_edge_features,
    y_to_A
)

# Set premium styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 1.0,
    'grid.color': '#f1f5f9',
    'grid.linewidth': 0.8,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'legend.frameon': True,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#e2e8f0',
})


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "linear_probe_all_params" not in cfg:
        raise ValueError("Missing 'linear_probe_all_params' in config.")

    args_dict = cfg["linear_probe_all_params"]
    args = SimpleNamespace(**args_dict)

    args.checkpoint = os.path.abspath(args.checkpoint)
    args.h5_path = os.path.abspath(args.h5_path)
    args.split_indices_path = os.path.abspath(args.split_indices_path)
    args.experiments_root = os.path.abspath(args.experiments_root)

    return args


# --- 1. Latent Extraction Functions ---




def extract_sns_embeddings(model, loader, device):
    """Extract per-human-node VAE embeddings."""
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode_nodes(batch)   # [N_total, latent_dim]
            keep = batch.action_label != IGNORE_LABEL
            if keep.any():
                all_z.append(z[keep].cpu())
                all_y.append(batch.action_label[keep].cpu())
    if not all_z:
        raise RuntimeError("No SNS human nodes found in this split.")
    return torch.cat(all_z, dim=0).numpy(), torch.cat(all_y, dim=0).long().numpy()



def reduce_dimensions(z, method="tsne", random_state=42):
    """Performs PCA or t-SNE reduction to 2D."""
    if method == "tsne":
        print(f"  Running t-SNE on feature dimension {z.shape[1]} (n_samples={z.shape[0]})...")
        perplexity = min(30.0, max(5.0, z.shape[0] / 3.0))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=1000)
    else:
        print(f"  Running PCA on feature dimension {z.shape[1]} (n_samples={z.shape[0]})...")
        reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(z)


# --- 3. Plotting Helper Functions ---

def save_and_clear_plot(out_path):
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved -> {out_path}")


def plot_binary_latent(z_2d, labels, title, label_names, out_path, colors=None):
    """Plots a single binary variable scatter plot."""
    if colors is None:
        colors = ['#64748b', '#6366f1']  # Muted slate and Indigo
    
    fig, ax = plt.subplots(figsize=(8, 7))
    for val, color, lbl in [(0, colors[0], label_names[0]), (1, colors[1], label_names[1])]:
        mask = (labels == val)
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=color, label=lbl, alpha=0.6, s=20,
            edgecolors='none'
        )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    ax.legend(markerscale=2, fontsize=10, loc='upper right', framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    save_and_clear_plot(out_path)


def plot_multiclass_latent(z_2d, labels, classes, title, out_path, colors=None):
    """Plots a multi-class variable scatter plot."""
    if colors is None:
        colors = ['#e11d48', '#2563eb', '#d97706', '#059669', '#64748b']  # Rosy, Royal Blue, Amber, Emerald, Slate
        
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, (name, color) in enumerate(zip(classes, colors)):
        mask = (labels == i)
        if mask.any():
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=color, label=name, alpha=0.6, s=18,
                edgecolors='none'
            )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    ax.legend(markerscale=2.5, fontsize=10, loc='upper right', framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    save_and_clear_plot(out_path)


def plot_combined_sean(z_2d, y, method, out_path):
    """Generates a combined 1x3 subplot figure for SEAN-T emotions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#64748b', '#e11d48']  # Slate vs Crimson
    
    for d, (ax, name) in enumerate(zip(axes, LABEL_NAMES)):
        labels = y[:, d].astype(int)
        for val, color, lbl in [(0, colors[0], 'Low'), (1, colors[1], 'High')]:
            mask = (labels == val)
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=color, label=lbl, alpha=0.5, s=12,
                edgecolors='none'
            )
        ax.set_title(f'{name.capitalize()}  (pos={labels.mean():.2%})', fontsize=12, fontweight='bold', pad=10)
        ax.legend(markerscale=2, fontsize=8, loc='upper right')
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
    fig.suptitle(f'SEAN-T Latent Space 2D Projection ({method.upper()})', fontsize=16, fontweight='bold', y=0.98, color='#0f172a')
    save_and_clear_plot(out_path)


# --- Main Orchestrator ---

def main():
    parser = argparse.ArgumentParser(description="Visualize latent space via t-SNE & PCA")
    parser.add_argument("--config", default="src/params/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default="plots", help="Output folder for saving plots")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument("--split", default="test", choices=["test", "train", "val", "all"], help="Which split to visualize")
    parser.add_argument("--hide-label-name", default=None, help="Specific SNS label name to hide (e.g. not_consider)")
    
    # Explicit boolean flags for hiding specific SNS action labels
    parser.add_argument("--hide-avoid", action="store_true", help="Hide 'avoid' action label in SNS plots")
    parser.add_argument("--hide-follow", action="store_true", help="Hide 'follow' action label in SNS plots")
    parser.add_argument("--hide-overtake", action="store_true", help="Hide 'overtake' action label in SNS plots")
    parser.add_argument("--hide-yield", action="store_true", help="Hide 'yield' action label in SNS plots")
    parser.add_argument("--hide-not_consider", action="store_true", help="Hide 'not_consider' action label in SNS plots")
    
    cmd_args = parser.parse_args()

    # Load configuration
    args = load_config(cmd_args.config)
    if cmd_args.checkpoint:
        args.checkpoint = cmd_args.checkpoint
    seed = cmd_args.seed
    seed_everything(seed)
    
    device = resolve_device(cmd_args.device if cmd_args.device else args.device)
    print(f"Using device: {device}")
    
    output_dir = os.path.abspath(cmd_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}\n")

    # Load VAE models and datasets
    print("Loading frozen HGNNCombinedVAE encoder...")
    encoder, robot_mean, robot_std, _ = load_frozen_encoder(
        checkpoint_path=args.checkpoint,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
        edge_transformer_layers=args.edge_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=device,
    )

    dataset = UnifiedGraphDataset(
        h5_path=args.h5_path,
        ssub=False,
        robot_mean=robot_mean,
        robot_std=robot_std,
    )
    splits = torch.load(args.split_indices_path, map_location="cpu", weights_only=False)
    
    # Resolve indices based on split argument
    if cmd_args.split == "all":
        split_indices = []
        for s in ["train", "val", "test"]:
            if s in splits:
                if torch.is_tensor(splits[s]):
                    split_indices.extend(splits[s].tolist())
                elif isinstance(splits[s], np.ndarray):
                    split_indices.extend(splits[s].tolist())
                else:
                    split_indices.extend(splits[s])
        print(f"Combining all splits. Total samples: {len(split_indices)}")
    else:
        split_indices = splits[cmd_args.split]
        if torch.is_tensor(split_indices):
            split_indices = split_indices.tolist()
        print(f"Using split '{cmd_args.split}'. Total samples: {len(split_indices)}")

    # ─── EXPERIMENT 1 & 2: SEAN-T (Graph Latent Space) ───
    print("\n" + "=" * 60)
    print("SEAN-T LATENT VISUALIZATION")
    print("=" * 60)
    
    sean_split = dataset.filter_by_source(split_indices, DATASET_SEAN)
    sean_loader = DataLoader(
        torch.utils.data.Subset(dataset, sean_split),
        batch_size=min(16, int(args.batch_size)),
        shuffle=False,
        num_workers=0
    )
    
    z_sean, y_sean = extract_sean_embeddings(
        encoder, sean_loader, device, int(args.latent_dim), args.sean_embedding_mode
    )
    print(f"SEAN-T embeddings shape: {z_sean.shape}")
    
    for method in ["tsne", "pca"]:
        z_2d_sean = reduce_dimensions(z_sean, method, random_state=seed)
        
        # Individual plots
        for d, name in enumerate(LABEL_NAMES):
            out_fn = os.path.join(output_dir, f"sean_{name}_{method}.png")
            plot_binary_latent(
                z_2d_sean, y_sean[:, d],
                f"SEAN-T {name.capitalize()} Latent Space ({method.upper()})",
                ["Low / Inactive", "High / Active"],
                out_fn
            )
            
        # Combined plot
        out_comb = os.path.join(output_dir, f"sean_combined_{method}.png")
        plot_combined_sean(z_2d_sean, y_sean, method, out_comb)

    # ─── NEW EXPERIMENT: SEAN-T Robot Node-level Latent Space ───
    print("\n" + "=" * 60)
    print("SEAN-T ROBOT NODE LATENT VISUALIZATION")
    print("=" * 60)
    
    z_sean_robot, y_sean_robot = extract_sean_robot_embeddings(
        encoder, sean_loader, device
    )
    print(f"SEAN-T robot node embeddings shape: {z_sean_robot.shape}")
    
    for method in ["tsne", "pca"]:
        z_2d_sean_robot = reduce_dimensions(z_sean_robot, method, random_state=seed)
        
        # Individual plots
        for d, name in enumerate(LABEL_NAMES):
            out_fn = os.path.join(output_dir, f"sean_robot_node_{name}_{method}.png")
            plot_binary_latent(
                z_2d_sean_robot, y_sean_robot[:, d],
                f"SEAN-T Robot Node {name.capitalize()} Latent Space ({method.upper()})",
                ["Low / Inactive", "High / Active"],
                out_fn
            )
            
        # Combined plot
        out_comb = os.path.join(output_dir, f"sean_robot_node_combined_{method}.png")
        plot_combined_sean(z_2d_sean_robot, y_sean_robot, method, out_comb)

    # ─── NEW EXPERIMENT: SEAN-T Follower Node-level Latent Space ───
    print("\n" + "=" * 60)
    print("SEAN-T FOLLOWER NODE LATENT VISUALIZATION")
    print("=" * 60)
    
    z_sean_follower, y_sean_follower = extract_sean_follower_embeddings(
        encoder, sean_loader, device
    )
    print(f"SEAN-T follower node embeddings shape: {z_sean_follower.shape}")
    
    for method in ["tsne", "pca"]:
        z_2d_sean_follower = reduce_dimensions(z_sean_follower, method, random_state=seed)
        
        # Individual plots
        for d, name in enumerate(LABEL_NAMES):
            out_fn = os.path.join(output_dir, f"sean_follower_node_{name}_{method}.png")
            plot_binary_latent(
                z_2d_sean_follower, y_sean_follower[:, d],
                f"SEAN-T Follower Node {name.capitalize()} Latent Space ({method.upper()})",
                ["Low / Inactive", "High / Active"],
                out_fn
            )
            
        # Combined plot
        out_comb = os.path.join(output_dir, f"sean_follower_node_combined_{method}.png")
        plot_combined_sean(z_2d_sean_follower, y_sean_follower, method, out_comb)

    # ─── EXPERIMENT 3 & 4: SNS (Node Action Latent Space) ───
    print("\n" + "=" * 60)
    print("SNS ACTION LATENT VISUALIZATION")
    print("=" * 60)
    
    eval_h5_path = getattr(args, "sns_eval_h5_path", None)
    if eval_h5_path and os.path.isfile(eval_h5_path) and cmd_args.split == "test":
        print(f"Loading SNS test dataset from: {eval_h5_path}")
        eval_dataset = UnifiedGraphDataset(
            h5_path=eval_h5_path,
            ssub=True,
            robot_mean=dataset.robot_mean,
            robot_std=dataset.robot_std,
        )
        sns_loader = DataLoader(eval_dataset, batch_size=min(16, int(args.batch_size)), shuffle=False)
    else:
        sns_loader = DataLoader(
            torch.utils.data.Subset(dataset, split_indices),
            batch_size=min(16, int(args.batch_size)),
            shuffle=False
        )
        
    z_sns, y_sns = extract_sns_embeddings(encoder, sns_loader, device)
    print(f"SNS node embeddings shape: {z_sns.shape}")
    
    # Filter out hidden SNS action labels if specified
    hidden_labels = []
    for lbl in ACTION_CLASSES:
        arg_attr = f"hide_{lbl}"
        if getattr(cmd_args, arg_attr, False):
            hidden_labels.append(lbl)
            
    if cmd_args.hide_label_name:
        for part in cmd_args.hide_label_name.split(","):
            part_clean = part.strip()
            if part_clean in ACTION_CLASSES and part_clean not in hidden_labels:
                hidden_labels.append(part_clean)
                
    if hidden_labels:
        hidden_indices = [ACTION_CLASSES.index(lbl) for lbl in hidden_labels]
        keep_mask = ~np.isin(y_sns, hidden_indices)
        z_sns = z_sns[keep_mask]
        y_sns = y_sns[keep_mask]
        print(f"Filtered SNS node embeddings shape (after hiding {hidden_labels}): {z_sns.shape}")
    
    for method in ["tsne", "pca"]:
        z_2d_sns = reduce_dimensions(z_sns, method, random_state=seed)
        
        # Multi-class plot (each action colored differently)
        out_fn = os.path.join(output_dir, f"sns_action_{method}.png")
        plot_multiclass_latent(
            z_2d_sns, y_sns, ACTION_CLASSES,
            f"SNS Human Action Node Latents ({method.upper()})",
            out_fn
        )
        # Combined plot (equal to multi-class plot for single multiclass variable)
        out_comb = os.path.join(output_dir, f"sns_combined_{method}.png")
        plot_multiclass_latent(
            z_2d_sns, y_sns, ACTION_CLASSES,
            f"SNS Human Action Latent Space Combined ({method.upper()})",
            out_comb
        )

    # ─── EXPERIMENT 5 & 6: Cocktail Party (Edge Detection Latent Space) ───
    print("\n" + "=" * 60)
    print("COCKTAIL PARTY EDGE LATENT VISUALIZATION")
    print("=" * 60)
    
    cp_dataset, cp_loaders = get_cocktail_loaders(
        args.cocktail_party_root, min(16, int(args.batch_size)), device, use_gnn_transforms=True
    )
    
    if cmd_args.split == "all":
        # Extract Option B (VAE Edge Latents) for all loaders and concatenate
        print("\nExtracting Option B (VAE Edge Latents 64D) representations for all splits...")
        z_cp_b_list, y_cp_b_list = [], []
        z_cp_a_list, y_cp_a_list = [], []
        for s in ["train", "val", "test"]:
            zb, yb = extract_cocktail_party_embeddings(encoder, cp_loaders[s], device, option_a=False)
            za, ya = extract_cocktail_party_embeddings(None, cp_loaders[s], device, option_a=True)
            z_cp_b_list.append(zb)
            y_cp_b_list.append(yb)
            z_cp_a_list.append(za)
            y_cp_a_list.append(ya)
        z_cp_b = np.concatenate(z_cp_b_list, axis=0)
        y_cp_b = np.concatenate(y_cp_b_list, axis=0)
        z_cp_a = np.concatenate(z_cp_a_list, axis=0)
        y_cp_a = np.concatenate(y_cp_a_list, axis=0)
    else:
        # Extract Option B (VAE Edge Latents) for specified split
        split_name = cmd_args.split
        print(f"\nExtracting Option B (VAE Edge Latents 64D) representations for split '{split_name}'...")
        z_cp_b, y_cp_b = extract_cocktail_party_embeddings(encoder, cp_loaders[split_name], device, option_a=False)
        print(f"\nExtracting Option A (Raw Edge PPF 4D) representations for split '{split_name}'...")
        z_cp_a, y_cp_a = extract_cocktail_party_embeddings(None, cp_loaders[split_name], device, option_a=True)

    print(f"Cocktail Party Option B embeddings shape: {z_cp_b.shape}")
    print(f"Cocktail Party Option A embeddings shape: {z_cp_a.shape}")
    
    # Subsample Cocktail Party to at most 5000 samples for fast and clean t-SNE / PCA visualization
    max_samples = 5000
    if z_cp_b.shape[0] > max_samples:
        print(f"\nSubsampling Cocktail Party samples from {z_cp_b.shape[0]} to {max_samples} for clean & fast visualization...")
        rng = np.random.default_rng(seed)
        indices = rng.choice(z_cp_b.shape[0], max_samples, replace=False)
        z_cp_b = z_cp_b[indices]
        y_cp_b = y_cp_b[indices]
        z_cp_a = z_cp_a[indices]
        y_cp_a = y_cp_a[indices]
        print(f"Subsampled Cocktail Party Option B embeddings shape: {z_cp_b.shape}")
        print(f"Subsampled Cocktail Party Option A embeddings shape: {z_cp_a.shape}")
    
    for method in ["tsne", "pca"]:
        # Option B (VAE Latents)
        z_2d_cp_b = reduce_dimensions(z_cp_b, method, random_state=seed)
        out_fn_b = os.path.join(output_dir, f"cocktail_party_vae_{method}.png")
        plot_binary_latent(
            z_2d_cp_b, y_cp_b,
            f"Cocktail Party VAE Edge Latents ({method.upper()})",
            ["Different Group", "Same Group"],
            out_fn_b,
            colors=['#64748b', '#10b981']  # Slate vs Emerald
        )
        
        # Option A (Raw Edge PPF)
        z_2d_cp_a = reduce_dimensions(z_cp_a, method, random_state=seed)
        out_fn_a = os.path.join(output_dir, f"cocktail_party_raw_{method}.png")
        plot_binary_latent(
            z_2d_cp_a, y_cp_a,
            f"Cocktail Party Raw Edge PPF Features ({method.upper()})",
            ["Different Group", "Same Group"],
            out_fn_a,
            colors=['#64748b', '#f59e0b']  # Slate vs Amber
        )
        
        # Combined plot per method showing Option A next to Option B
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        colors_a = ['#64748b', '#f59e0b']
        colors_b = ['#64748b', '#10b981']
        
        # Option A Plot
        for val, color, lbl in [(0, colors_a[0], "Diff Group"), (1, colors_a[1], "Same Group")]:
            mask = (y_cp_a == val)
            axes[0].scatter(z_2d_cp_a[mask, 0], z_2d_cp_a[mask, 1], c=color, label=lbl, alpha=0.5, s=15, edgecolors='none')
        axes[0].set_title("Option A: Raw Edge PPF Features", fontsize=12, fontweight='bold', pad=10)
        axes[0].legend(markerscale=2, fontsize=9)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
        axes[0].spines['left'].set_color('#cbd5e1'); axes[0].spines['bottom'].set_color('#cbd5e1')
        
        # Option B Plot
        for val, color, lbl in [(0, colors_b[0], "Diff Group"), (1, colors_b[1], "Same Group")]:
            mask = (y_cp_b == val)
            axes[1].scatter(z_2d_cp_b[mask, 0], z_2d_cp_b[mask, 1], c=color, label=lbl, alpha=0.5, s=15, edgecolors='none')
        axes[1].set_title("Option B: VAE Latent Spaces", fontsize=12, fontweight='bold', pad=10)
        axes[1].legend(markerscale=2, fontsize=9)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
        axes[1].spines['left'].set_color('#cbd5e1'); axes[1].spines['bottom'].set_color('#cbd5e1')
        
        fig.suptitle(f"Cocktail Party Edge Dimensionality Reduction ({method.upper()})", fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
        out_comb = os.path.join(output_dir, f"cocktail_party_combined_{method}.png")
        save_and_clear_plot(out_comb)

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"All output files have been saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
