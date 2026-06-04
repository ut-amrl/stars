import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from features.mlp_conditional_factored_vae_sean import OccupancyVAE, NearbyVAE, nearby_chamfer_distance
except ImportError:
    OccupancyVAE = None
    NearbyVAE = None
    nearby_chamfer_distance = None

POSITION_MEAN = np.array([0.42840074, 0.06433814])
POSITION_STD  = np.sqrt(np.array([25.15945677, 20.42660306]))


def _nearby_to_numpy(nearby_raw):
    """Convert nearby array (torch or numpy) to numpy float32 without deprecation warnings."""
    if torch.is_tensor(nearby_raw):
        return nearby_raw.cpu().numpy().astype(np.float32)
    return np.asarray(nearby_raw, dtype=np.float32)


def _nearby_to_tensor(nearby_raw):
    """Convert nearby array to torch float32 tensor without copy warnings when already a tensor."""
    if torch.is_tensor(nearby_raw):
        return nearby_raw.detach().to(torch.float32)
    return torch.tensor(nearby_raw, dtype=torch.float32)


def _sigmoid(x):
    """Stable sigmoid to avoid overflow in exp for large logits."""
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def load_occupancy_vae(model_path: str, device: str = "cpu") -> OccupancyVAE:
    """Load a trained OccupancyVAE from a min_val_loss.pth checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file (e.g. KNQJ0SJZ_occ_vae_min_val_loss.pth)
        device: torch device string
    Returns:
        OccupancyVAE model in eval mode on the specified device
    """
    checkpoint_dict = torch.load(model_path, map_location=device, weights_only=False)
    model = OccupancyVAE(**checkpoint_dict["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.eval()
    print(f"Loaded OccupancyVAE from {model_path}")
    print(f"  kwargs: {checkpoint_dict['model_kwargs']}")
    print(f"  epoch: {checkpoint_dict['epoch']}, val_loss: {checkpoint_dict['val_loss']:.4f}")
    return model


def load_nearby_vae(model_path: str, device: str = "cpu") -> NearbyVAE:
    """Load a trained NearbyVAE from a min_val_loss.pth checkpoint."""
    checkpoint_dict = torch.load(model_path, map_location=device, weights_only=False)
    model = NearbyVAE(**checkpoint_dict["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.eval()
    print(f"Loaded NearbyVAE from {model_path}")
    print(f"  kwargs: {checkpoint_dict['model_kwargs']}")
    print(f"  epoch: {checkpoint_dict['epoch']}, val_loss: {checkpoint_dict['val_loss']:.4f}")
    return model


def _draw_scene(ax, occupancy_grid, goal_positions_t, nearby_positions_t,
                nearby_orientations_t, follower_positions_t, follower_orientations_t,
                extent=(-3.6, 3.6, -3.6, 3.6), vmin=0, vmax=100, dot_size=40):
    """Render a single timestep's scene onto a matplotlib Axes."""
    ax.imshow(occupancy_grid, extent=extent, vmin=vmin, vmax=vmax)
    ax.scatter(0, 0, color='blue', s=dot_size, zorder=5)
    ax.arrow(0, 0, 0.8, 0, color='blue', head_width=0.2, head_length=0.2, zorder=6)
    ax.scatter(goal_positions_t[0], goal_positions_t[1], color='green', marker="s", s=dot_size, zorder=5)
    for i in range(nearby_positions_t.shape[0]):
        if nearby_orientations_t[i, 0]**2 + nearby_orientations_t[i, 1]**2 > 0:
            ax.scatter(nearby_positions_t[i, 0], nearby_positions_t[i, 1], color='grey', s=dot_size, zorder=5)
            ax.arrow(nearby_positions_t[i, 0], nearby_positions_t[i, 1],
                     0.8 * nearby_orientations_t[i, 0], 0.8 * nearby_orientations_t[i, 1],
                     color='grey', head_width=0.2, head_length=0.2, zorder=6)
    ax.scatter(follower_positions_t[0], follower_positions_t[1], color='red', s=dot_size, zorder=5)
    ax.arrow(follower_positions_t[0], follower_positions_t[1],
             0.8 * follower_orientations_t[0], 0.8 * follower_orientations_t[1],
             color='red', head_width=0.2, head_length=0.2, zorder=6)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])


## Inside SEAN Dataset, there are folders of examples, inside each one are several temporal_*.pt
## Each one is a roughly 8 second trajectory

## x.keys()
## dict_keys(['physical', 'spatial', 'facial', 'viz', 'other', 
##            'y', 'graph', 'after_behavior_change', 'final_label',
##            'have_label', 'true_label'])
##
## x["physical"]["map_resnet18"] -- seems some type of latent representation of the map?
## x["facial"].keys() --> dict_keys(['eye', 'lip', 'head', 'gaze'])
## x["viz"].keys() --> dict_keys(['map', 'eye_data_validata_bit_mask_left', 'eye_data_validata_bit_mask_right', 'eye_data_validata_bit_mask_combined', 'convergence_distance_validity', 'convergence_distance_mm', 'tracking_improvements', 'tracking_improvements_length', 'gaze_direction_local', 'gaze_direction_global', 'gaze_origin_local', 'gaze_origin_global'])
## x["other"].keys() --> dict_keys(['behavior', 'time_elapsed', 'time_remaining'])
## x["y"] --> tensor with 3 elements
## x["graph"] --> torch_geometric.data.data.Data (keys are: ['x', 'y', 'edge_attr', 'edge_indices'])
## x["after_behavior_change"] --> bool
## x["final_label"] --> numpy.bool
## x["have_label"] --> numpy.bool
## x["true_label"] --> numpy array with 3 elements
##
## x["spatial"].keys() --> dict_keys(['goal', 'robot', 'follower', 'nearby'])
##
## x["spatial"]["goal"] --> (40, 2) array        Location of the goal wrt to the robot's position?
## x["spatial"]["robot"] --> (40, 2) array       No, seems to be cos sin values in some global frame?
## x["spatial"]["follower"] --> (40, 4) array    Position and cos sin of follower.
##                                               index 2,3 is cos sin
##                                               index 0 seems to x-axis, representing how far behind the robot the person is
##                                               index 1 seems to be y-axis, representing how far side-to-side the person is
## x["spatial"]["nearby"] --> (40, 16, 4) array  Positons and cos sin of nearby people?
##
## NOTE: x["spatial"]["nearby"] == x["graph"]["x"]
##       x["graph"]["y"] == x["true_label"]
## 
## x["other"]["behavior"] --> (40, 1) array      Behavior label at each timestep
## x["viz"]["map"] is a list, each element is the occupancy grid at that timestep.
##         Each element is a 72 x 72 numpy array. occupancy grid?
##         Empty space seems to be 100, occupied seems to be 0. 
##         Robot seems to be located at center (35, 35)
##
## Trajectory is: occupancy grid + robot + goal + nearby people

def visualize_sean_trajectory(prefix: str, idx_list: list[int]):
    for idx in idx_list:
        print(f"Loading temporal_{idx}... from {prefix}")
        x = torch.load(prefix + f'temporal_{idx}.pt', weights_only=False)

        ## Each row represents the orientation of the robot, e.g. cos \theta, sin \theta
        robot_orientation = x["spatial"]["robot"]
        goal_positions = x["spatial"]["goal"]
        follower_positions = x["spatial"]["follower"][:, :2]
        follower_orientations = x["spatial"]["follower"][:, 2:]
        nearby_positions = x["spatial"]["nearby"][:, :, :2]
        nearby_orientations = x["spatial"]["nearby"][:, :, 2:]
        occupancy_grids = x["viz"]["map"]
        ## Size of the map should be 7.2 by 7.2 in the plot
        dot_size = 40

        for t in range(40):
            print(f"Processing frame {40*idx+t}...")
            ## Occupancy grid
            plt.imshow(occupancy_grids[t], extent=[-3.6, 3.6, -3.6, 3.6])
            ## Robot is in the center, facing in the +x direction
            plt.scatter(0, 0, color='blue', s=dot_size, zorder=5)
            plt.arrow(0, 0, 0.8, 0, color='blue', head_width=0.2, head_length=0.2, zorder=6)
            ## Goal location wrt to robot
            plt.scatter(goal_positions[t, 0], goal_positions[t, 1], color='green', marker="s", s=dot_size, zorder=5)
            for i in range(16):
                if nearby_orientations[t, i, 0]**2 +nearby_orientations[t, i, 1]**2 > 0:
                    plt.scatter(nearby_positions[t, i, 0], nearby_positions[t, i, 1], color='grey', s=dot_size, zorder=5)
                    plt.arrow(nearby_positions[t, i, 0], nearby_positions[t, i, 1], 
                            0.8 * nearby_orientations[t, i, 0], 0.8 * nearby_orientations[t, i, 1], 
                            color='grey', head_width=0.2, head_length=0.2, zorder=6)
            ## Follower location, and orientation
            plt.scatter(follower_positions[t, 0], follower_positions[t, 1], color='red', s=dot_size, zorder=5)
            plt.arrow(follower_positions[t, 0], follower_positions[t, 1], 
                    0.8 * follower_orientations[t, 0], 0.8 * follower_orientations[t, 1], 
                    color='red', head_width=0.2, head_length=0.2, zorder=6)
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            print(f"  ... Saving frame {40*idx+t}...")
            plt.savefig(f"frames/map_frames_{40*idx+t:05d}.png", bbox_inches='tight')
            plt.close()

def visualize_predicted_and_ground_truth_trajectory(
        prefix: str, idx_list: list[int],
        occ_model: OccupancyVAE, nearby_model: NearbyVAE,
        output_dir: str = "frames",
        chamfer_output_path: str | None = None,
):
    """Side-by-side: ground truth (left) vs VAE-reconstructed (right).

    Left panel is identical to visualize_sean_trajectory.
    Right panel replaces the occupancy grid and nearby people with
    reconstructions from the OccupancyVAE and NearbyVAE respectively.
    Goal and follower are shown identically on both sides (no VAE for those).

    If chamfer_output_path is set, per-frame chamfer distance and MSE are
    computed and saved to that path (same format as compute_and_pickle_chamfer).
    """
    os.makedirs(output_dir, exist_ok=True)
    occ_device = next(occ_model.parameters()).device
    nearby_device = next(nearby_model.parameters()).device

    all_t = [] if chamfer_output_path else None
    all_d = [] if chamfer_output_path else None
    all_mse = [] if chamfer_output_path else None

    for idx in idx_list:
        print(f"Loading temporal_{idx}... from {prefix}")
        x = torch.load(prefix + f'temporal_{idx}.pt', weights_only=False)

        goal_positions = x["spatial"]["goal"]
        follower_positions = x["spatial"]["follower"][:, :2]
        follower_orientations = x["spatial"]["follower"][:, 2:]
        nearby_raw = x["spatial"]["nearby"]                    # (40, 16, 4)
        nearby_positions = nearby_raw[:, :, :2]
        nearby_orientations = nearby_raw[:, :, 2:]
        raw_grids = x["viz"]["map"]                            # list of (72,72), values 0-100

        occ_tensor = torch.tensor(np.array(raw_grids), dtype=torch.float32) / 100.0
        occ_tensor = occ_tensor.unsqueeze(1)                   # (T,1,72,72)

        nearby_np = _nearby_to_numpy(nearby_raw)     # (T,16,4)
        nearby_norm = nearby_np.copy()
        nearby_norm[:, :, 0:2] = (nearby_np[:, :, 0:2] - POSITION_MEAN) / (POSITION_STD + 1e-8)
        valid_mask = nearby_np[:, :, 2:]**2
        valid_mask = valid_mask.sum(axis=-1) > 0.5             # (T,16)
        nearby_norm[~valid_mask] = 0.0
        nearby_tensor = torch.tensor(nearby_norm, dtype=torch.float32)  # (T,16,4) normalized

        with torch.no_grad():
            recon_occ, _, _ = occ_model(occ_tensor.to(occ_device))
            recon_nearby_mu, _ = nearby_model.encode(nearby_tensor.to(nearby_device))
            recon_nearby_raw = nearby_model.decode(recon_nearby_mu)

        recon_occ = recon_occ.cpu().numpy()                    # (T,1,72,72) in [0,1]
        recon_nearby_raw = recon_nearby_raw.cpu().numpy()      # (T,16,5)

        recon_nearby_pos = recon_nearby_raw[:, :, :2] * (POSITION_STD + 1e-8) + POSITION_MEAN
        recon_nearby_ori = recon_nearby_raw[:, :, 2:4]
        recon_exist_prob = _sigmoid(recon_nearby_raw[:, :, 4])

        for t in range(occ_tensor.shape[0]):
            frame_id = 40 * idx + t
            print(f"Processing frame {frame_id}...")

            fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 7))

            _draw_scene(ax_gt, raw_grids[t],
                        goal_positions[t], nearby_positions[t], nearby_orientations[t],
                        follower_positions[t], follower_orientations[t])
            ax_gt.set_title(f"Ground Truth  (t={t})")

            pred_nearby_ori_t = recon_nearby_ori[t].copy()
            pred_nearby_ori_t[recon_exist_prob[t] < 0.5] = 0.0
            _draw_scene(ax_pred, recon_occ[t, 0] * 100.0,
                        goal_positions[t], recon_nearby_pos[t], pred_nearby_ori_t,
                        follower_positions[t], follower_orientations[t])
            ax_pred.set_title(f"Predicted  (t={t})")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pred_vs_gt_{frame_id:05d}.png"), bbox_inches='tight')
            plt.close(fig)

        print(f"Saved {occ_tensor.shape[0]} frames for temporal_{idx} to {output_dir}/")

        if chamfer_output_path is not None:
            T = nearby_raw.shape[0]
            recon_4 = np.zeros((T, 16, 4), dtype=np.float32)
            recon_4[:, :, :2] = recon_nearby_pos
            recon_4[:, :, 2:4] = recon_nearby_ori
            recon_4[recon_exist_prob < 0.5] = 0.0
            gt_np = _nearby_to_numpy(nearby_raw)
            gt_tensor = _nearby_to_tensor(nearby_raw)
            recon_tensor = torch.tensor(recon_4, dtype=torch.float32)
            with torch.no_grad():
                d = nearby_chamfer_distance(gt_tensor, recon_tensor)
            d_np = d.cpu().numpy()
            mse_np = np.mean((gt_np - recon_4) ** 2, axis=(1, 2))
            all_t.append(np.arange(T, dtype=np.int64))
            all_d.append(d_np)
            all_mse.append(mse_np)

    if chamfer_output_path is not None:
        t_concat = np.concatenate(all_t)
        d_concat = np.concatenate(all_d)
        mse_concat = np.concatenate(all_mse)
        parent = os.path.dirname(chamfer_output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(chamfer_output_path, "wb") as f:
            pickle.dump({"t": t_concat, "d": d_concat, "mse": mse_concat}, f)
        print(f"Saved chamfer and MSE data to {chamfer_output_path} (len={len(t_concat)})")


def compute_and_pickle_chamfer(
    prefix: str, idx_list: list[int],
    nearby_model: NearbyVAE,
    output_path: str = "chamfer.pkl",
):
    """Compute per-frame chamfer distance and MSE between GT and reconstructed nearby, pickle to file.

    For each temporal_*.pt in idx_list: load data, run through NearbyVAE, then
    compute nearby_chamfer_distance(gt, recon) and per-frame MSE. Accumulates t, d, and mse
    across all trajectories and saves {'t': array, 'd': array, 'mse': array} to output_path.
    """
    device = next(nearby_model.parameters()).device
    all_t = []
    all_d = []
    all_mse = []

    for idx in idx_list:
        print(f"Loading temporal_{idx}... from {prefix}")
        x = torch.load(prefix + f'temporal_{idx}.pt', weights_only=False)

        nearby_raw = x["spatial"]["nearby"]  # (T, 16, 4)
        T = nearby_raw.shape[0]

        nearby_np = _nearby_to_numpy(nearby_raw)
        nearby_norm = nearby_np.copy()
        nearby_norm[:, :, 0:2] = (nearby_np[:, :, 0:2] - POSITION_MEAN) / (POSITION_STD + 1e-8)
        valid_mask = (nearby_np[:, :, 2:] ** 2).sum(axis=-1) > 0.5
        nearby_norm[~valid_mask] = 0.0
        nearby_tensor = torch.tensor(nearby_norm, dtype=torch.float32)

        with torch.no_grad():
            recon_nearby_mu, _ = nearby_model.encode(nearby_tensor.to(device))
            recon_nearby_raw = nearby_model.decode(recon_nearby_mu)

        recon_nearby_raw = recon_nearby_raw.cpu().numpy()  # (T, 16, 5)
        recon_nearby_pos = recon_nearby_raw[:, :, :2] * (POSITION_STD + 1e-8) + POSITION_MEAN
        recon_nearby_ori = recon_nearby_raw[:, :, 2:4]
        recon_exist_prob = _sigmoid(recon_nearby_raw[:, :, 4])

        # Build (T, 16, 4) recon tensor; zero out rows where exist < 0.5
        recon_4 = np.zeros((T, 16, 4), dtype=np.float32)
        recon_4[:, :, :2] = recon_nearby_pos
        recon_4[:, :, 2:4] = recon_nearby_ori
        recon_4[recon_exist_prob < 0.5] = 0.0

        gt_tensor = _nearby_to_tensor(nearby_raw)
        recon_tensor = torch.tensor(recon_4, dtype=torch.float32)

        with torch.no_grad():
            d = nearby_chamfer_distance(gt_tensor, recon_tensor)  # (T,)
        d_np = d.cpu().numpy()
        # Per-frame MSE over (16, 4) between GT and recon (both in raw space)
        gt_np = _nearby_to_numpy(nearby_raw)
        mse_np = np.mean((gt_np - recon_4) ** 2, axis=(1, 2))  # (T,)
        t_np = np.arange(T, dtype=np.int64)

        all_t.append(t_np)
        all_d.append(d_np)
        all_mse.append(mse_np)

    t_concat = np.concatenate(all_t)
    d_concat = np.concatenate(all_d)
    mse_concat = np.concatenate(all_mse)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"t": t_concat, "d": d_concat, "mse": mse_concat}, f)
    print(f"Saved chamfer and MSE data (t, d, mse) to {output_path} (len={len(t_concat)})")


def plot_chamfer_from_pickles(
    pickle_paths: list[str],
    labels: list[str],
    output_path: str = "chamfer_plot.png",
):
    """Load N pickle files (each with 't', 'd', 'mse'), plot chamfer and MSE vs time."""
    if len(pickle_paths) != len(labels):
        raise ValueError("pickle_paths and labels must have the same length")

    fig, (ax_chamfer, ax_mse) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for path, label in zip(pickle_paths, labels):
        with open(path, "rb") as f:
            data = pickle.load(f)
        t, d, mse = data["t"], data["d"], data["mse"]
        ax_chamfer.plot(t, d, label=label)
        ax_mse.plot(t, mse, label=label)

    ax_chamfer.set_ylabel("Chamfer Distance")
    ax_chamfer.set_title("Chamfer Distance vs Time")
    ax_chamfer.legend()
    ax_chamfer.grid(True, alpha=0.3)

    ax_mse.set_xlabel("Frame")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("MSE vs Time (reference)")
    ax_mse.legend()
    ax_mse.grid(True, alpha=0.3)

    plt.tight_layout()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved chamfer plot to {output_path}")


def visualize_occupancy_reconstruction(prefix: str, idx: int, model: OccupancyVAE, output_dir: str = "frames"):
    """Show ground-truth and reconstructed occupancy grids side-by-side.

    Raw occupancy values (0=occupied, 100=free) are normalised to [0,1]
    before being fed through the VAE, matching the training pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    print(f"Loading temporal_{idx}... from {prefix}")
    x = torch.load(prefix + f'temporal_{idx}.pt', weights_only=False)
    raw_grids = x["viz"]["map"]  # list of (72,72) arrays, values in [0,100]

    occ_tensor = torch.tensor(np.array(raw_grids), dtype=torch.float32) / 100.0  # (T,72,72) in [0,1]
    occ_tensor = occ_tensor.unsqueeze(1)  # (T,1,72,72)

    with torch.no_grad():
        recon, mu, logvar = model(occ_tensor.to(device))
    recon = recon.cpu().numpy()  # (T,1,72,72) in [0,1]

    extent = [-3.6, 3.6, -3.6, 3.6]
    for t in range(occ_tensor.shape[0]):
        frame_id = 40 * idx + t
        print(f"Processing frame {frame_id}...")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(raw_grids[t], extent=extent, vmin=0, vmax=100)
        axes[0].set_title(f"Ground Truth  (t={t})")

        axes[1].imshow(recon[t, 0] * 100.0, extent=extent, vmin=0, vmax=100)
        axes[1].set_title(f"Reconstruction  (t={t})")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"occ_recon_{frame_id:05d}.png"), bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {occ_tensor.shape[0]} frames to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAN trajectory / occupancy visualizer")
    parser.add_argument("--mode", choices=["trajectory", "occ_recon", "pred_vs_gt", "chamfer_pickle", "chamfer_plot"], default="pred_vs_gt",
                        help="Visualization mode")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Path to folder containing temporal_*.pt files (include trailing /)")
    parser.add_argument("--idx", type=int, nargs="+", default=[0],
                        help="temporal_*.pt index(es) to visualise")
    parser.add_argument("--occ_model", type=str, default=None,
                        help="Path to OccupancyVAE checkpoint (.pth)")
    parser.add_argument("--nearby_model", type=str, default=None,
                        help="Path to NearbyVAE checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="frames")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for chamfer_pickle (.pkl) or chamfer_plot (.png)")
    parser.add_argument("--chamfer_output", type=str, default=None,
                        help="If set in pred_vs_gt mode, also save chamfer and MSE to this .pkl path")
    parser.add_argument("--pickle_files", type=str, nargs="+", default=None,
                        help="Paths to chamfer pickle files (for chamfer_plot)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each pickle file (for chamfer_plot)")
    args = parser.parse_args()

    if args.mode == "trajectory":
        if args.prefix is None:
            parser.error("--prefix is required for trajectory mode")
        visualize_sean_trajectory(prefix=args.prefix, idx_list=args.idx)

    elif args.mode == "occ_recon":
        if args.occ_model is None:
            parser.error("--occ_model is required for occ_recon mode")
        if args.prefix is None:
            parser.error("--prefix is required for occ_recon mode")
        occ_model = load_occupancy_vae(args.occ_model, device=args.device)
        for i in args.idx:
            visualize_occupancy_reconstruction(args.prefix, i, model=occ_model, output_dir=args.output_dir)

    elif args.mode == "pred_vs_gt":
        if args.occ_model is None or args.nearby_model is None:
            parser.error("--occ_model and --nearby_model are both required for pred_vs_gt mode")
        if args.prefix is None:
            parser.error("--prefix is required for pred_vs_gt mode")
        occ_model = load_occupancy_vae(args.occ_model, device=args.device)
        nearby_model = load_nearby_vae(args.nearby_model, device=args.device)
        visualize_predicted_and_ground_truth_trajectory(
            prefix=args.prefix, idx_list=args.idx,
            occ_model=occ_model, nearby_model=nearby_model,
            output_dir=args.output_dir,
            chamfer_output_path=args.chamfer_output)

    elif args.mode == "chamfer_pickle":
        if args.nearby_model is None:
            parser.error("--nearby_model is required for chamfer_pickle mode")
        if args.prefix is None:
            parser.error("--prefix is required for chamfer_pickle mode")
        output_path = args.output_path if args.output_path is not None else "chamfer.pkl"
        nearby_model = load_nearby_vae(args.nearby_model, device=args.device)
        compute_and_pickle_chamfer(
            prefix=args.prefix, idx_list=args.idx,
            nearby_model=nearby_model,
            output_path=output_path,
        )

    elif args.mode == "chamfer_plot":
        if args.pickle_files is None or args.labels is None:
            parser.error("--pickle_files and --labels are required for chamfer_plot mode")
        if len(args.pickle_files) != len(args.labels):
            parser.error("--pickle_files and --labels must have the same length")
        output_path = args.output_path if args.output_path is not None else "chamfer_plot.png"
        plot_chamfer_from_pickles(
            pickle_paths=args.pickle_files,
            labels=args.labels,
            output_path=output_path,
        )