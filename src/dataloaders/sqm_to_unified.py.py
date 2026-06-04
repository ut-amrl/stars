import argparse

import h5py
import numpy as np

from social_nav_converter_unified import (
    IGNORE_LABEL,
    NODE_FEATURE_DIM,
    UNIFIED_TIMESTEPS,
    write_combined_sample,
)

DATASET_SWM = 3

PASSTHROUGH_DATASETS = (
    "target_xy",
    "target_valid",
    "future_xy",
    "future_valid",
    "baseline_xy",
    "obs_xy",
    "node_ids",
    "people_ids",
    "track_ids",
)


def list_segment_group_keys(handle):
    """Collect group paths that look like materialized segment graphs."""
    keys = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "node_type" in obj and "edge_attr" in obj:
            keys.append(name)

    handle.visititems(visit)
    return sorted(keys)


def convert(swm_h5, output_path):
    n_written = 0
    n_skipped = 0
    n_not_full = 0
    n_min = n_max = None
    e_min = e_max = None

    with h5py.File(swm_h5, "r") as src, h5py.File(output_path, "w") as out:
        group_keys = list_segment_group_keys(src)
        print(f"[swm->unified] found {len(group_keys)} segment groups in {swm_h5}")

        for key in group_keys:
            grp = src[key]
            try:
                node_type = grp["node_type"][:].astype(np.int64)
                edge_index = grp["edge_index"][:].astype(np.int64)
                edge_attr = grp["edge_attr"][:].astype(np.float32)
                episode_key = grp.attrs.get("episode_key", key)
                if isinstance(episode_key, bytes):
                    episode_key = episode_key.decode("utf-8")

                n_nodes = int(node_type.shape[0])
                if n_nodes < 2 or edge_index.shape[0] != 2:
                    n_skipped += 1
                    continue

                # Shape contract required by HGNNCombinedVAE.
                assert edge_attr.shape[1:] == (UNIFIED_TIMESTEPS, 8), \
                    f"{key}: edge_attr {edge_attr.shape} != (E,{UNIFIED_TIMESTEPS},8)"
                assert edge_attr.shape[0] == edge_index.shape[1], \
                    f"{key}: edge_attr/edge_index count mismatch"
                if edge_index.size and int(edge_index.max()) >= n_nodes:
                    raise AssertionError(f"{key}: edge_index references node >= N")

                # Sanity guard (not fatal): source declares edge_strategy='full'.
                e_count = int(edge_index.shape[1])
                if e_count != n_nodes * (n_nodes - 1):
                    n_not_full += 1

                node_features = np.zeros((n_nodes, NODE_FEATURE_DIM), dtype=np.float32)
                valid_mask = np.ones((n_nodes, UNIFIED_TIMESTEPS), dtype=np.float32)
                emotion_label = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
                action_label = np.full(n_nodes, IGNORE_LABEL, dtype=np.int64)

                write_combined_sample(
                    out, n_written,
                    node_features, node_type, valid_mask,
                    edge_index, edge_attr,
                    emotion_label, action_label,
                    DATASET_SWM, f"swm/{episode_key}",
                )

                # Preserve forecasting fields + id arrays for downstream tasks.
                grp_out = out[f"sample_{n_written}"]
                for name in PASSTHROUGH_DATASETS:
                    if name in grp:
                        grp.copy(name, grp_out)

                # Preserve all source provenance attrs under a `src_` prefix.
                for attr_key, attr_val in grp.attrs.items():
                    grp_out.attrs[f"src_{attr_key}"] = attr_val

                n_min = n_nodes if n_min is None else min(n_min, n_nodes)
                n_max = n_nodes if n_max is None else max(n_max, n_nodes)
                e_min = e_count if e_min is None else min(e_min, e_count)
                e_max = e_count if e_max is None else max(e_max, e_count)
                n_written += 1
            except Exception as exc:
                print(f"  SKIP '{key}': {exc}")
                n_skipped += 1

    print(f"\n[swm->unified] done. written={n_written} skipped={n_skipped}")
    if n_written:
        print(f"  nodes per graph: min={n_min} max={n_max}")
        print(f"  edges per graph: min={e_min} max={e_max}  (fully connected = N*(N-1))")
        print(f"  non-fully-connected graphs: {n_not_full}")
    print(f"  saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SWM speedway graph h5 to the unified group format.")
    parser.add_argument(
        "--swm_h5",
        default="/robodata/speedway/SWM_speedway/cpt_graph_forecasting_10pct.h5",
        help="Input SWM materialized-graph h5 file.")
    parser.add_argument(
        "--output", default="/robodata/speedway/SWM_speedway/cpt_graph_forecasting_10pct_unified.h5",
        help="Output unified-format h5 file.")
    args = parser.parse_args()

    convert(args.swm_h5, args.output)
