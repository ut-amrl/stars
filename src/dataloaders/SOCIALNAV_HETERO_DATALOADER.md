# SocialNav-SUB Hetero Dataloader — Field Reference

This document describes every field in the `HeteroData` produced by
`SocialNavGraphDatasetV1` (file: `socialnav_hetero_graph_dataset_v1.py`),
the hetero PyG dataloader for the SocialNav-SUB dataset.

The HDF5 input is produced by `socialnav_to_hdf5_v3.py convert` over
`socialnavsub_model_labels_v2/human_oracle/` (60 real episodes, ground-truth
human-annotated labels). Generated file: `socialnav_human_oracle_v3.h5`.

## Top-level dataset facts

| Property | Value | Notes |
|---|---|---|
| Source | `socialnavsub_model_labels_v2/human_oracle/` | Real human annotations only (Strategy A) |
| Number of graphs | 60 | One per episode |
| Timesteps T | 10 | Constant across all episodes; from `n_waypoints_ahead=10` |
| Humans per graph P | 1–13 (avg 6.2) | Variable — PyG batches handle this natively |
| Node types | 2 | `robot`, `human` |
| Edge types | 3 directed | `(robot→human)`, `(human→robot)`, `(human→human)` |
| Total per-human labels | 372 | sum of P across 60 graphs |
| Action classes | 5 | `avoid (0), follow (1), overtake (2), yield (3), not_consider (4)` |
| Label distribution | not_consider=294, avoid=43, follow=29, yield=6, overtake=0 | Matches `generation_summary.json` |

## Per-graph `HeteroData` fields

### Node features

| Field | Shape | Dtype | Role | Domain | Purpose |
|---|---|---|---|---|---|
| `data['robot'].x` | `[1, 30]` | float32 | input | `[x, y, θ]` per timestep, robot's initial frame | Flattened robot trajectory `(T*3=30)`; primary input to robot encoder. |
| `data['robot'].traj` | `[1, T=10, 3]` | float32 | input (alt) | same as above, temporal form | Same data unflattened — useful if a model wants temporal modeling on robot. |
| `data['robot'].goal` | `[1, 2]` | float32 | input | `(gx, gy)` meters, robot's initial frame | Robot's destination/goal point — strong prior for the action it will take toward each human. |
| `data['human'].x` | `[P, 30]` | float32 | input | `[x, y, θ]` per timestep, robot's initial frame | Flattened per-human trajectory; primary input to human encoder. |
| `data['human'].traj` | `[P, T=10, 3]` | float32 | input (alt) | same as above | Temporal form of human trajectories. |

> Coordinates are all relative to the robot's initial pose (frame at t=0). θ is in radians.

### Labels (ground truth)

| Field | Shape | Dtype | Role | Domain | Purpose |
|---|---|---|---|---|---|
| `data['human'].y` | `[P]` | int64 | gt | `{0,1,2,3,4}` | Per-human action class (`avoid/follow/overtake/yield/not_consider`); used by the linear probe / multiclass head. |

> Robot has no `.y`. The HDF5's `action_label[0] = -100` (IGNORE_LABEL) is dropped; only humans carry labels.

### Edges (per directed type)

For each `et ∈ {(robot,'to',human), (human,'to',robot), (human,'to','human')}`:

| Field | Shape | Dtype | Role | Domain | Purpose |
|---|---|---|---|---|---|
| `data[et].edge_index` | `[2, E_t]` | int64 | structural | local node indices per type | COO format — row 0 = src local idx, row 1 = dst local idx. |
| `data[et].edge_attr` | `[E_t, T=10, 8]` | float32 | input | see edge feature dims below | Temporal SE(2) relative-pose features per edge. |

Edge counts per graph (fully connected directed, no self-loops):
- `(robot→human)`:  `E = P`
- `(human→robot)`:  `E = P`
- `(human→human)`:  `E = P*(P-1)`
- Total: `(1+P)*P` (verified at runtime).

#### Edge feature dimensions (per timestep, the trailing 8)

Computed by `se2_relative_pose` in `socialnav_to_hdf5_v2.py`:

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `rel_x` | meters | Destination x in source's local frame at time t |
| 1 | `rel_y` | meters | Destination y in source's local frame at time t |
| 2 | `cos_rel` | `[-1, 1]` | Cosine of dst heading expressed in src's local heading frame |
| 3 | `sin_rel` | `[-1, 1]` | Sine of dst heading in src's local heading frame |
| 4 | `d_rel_x` | meters/step | `rel_x[t] - rel_x[t-1]` (velocity-like; 0 at t=0) |
| 5 | `d_rel_y` | meters/step | `rel_y[t] - rel_y[t-1]` (velocity-like; 0 at t=0) |
| 6 | `d_cos`   | `[-2, 2]` | `cos_rel[t] - cos_rel[t-1]` (heading delta proxy; 0 at t=0) |
| 7 | `d_sin`   | `[-2, 2]` | `sin_rel[t] - sin_rel[t-1]` (heading delta proxy; 0 at t=0) |

> All 3 edge types share the same 8D feature schema. The hetero distinction is structural (which nodes are connected), not in the edge feature space.

### Per-graph metadata (plain Python attributes on `HeteroData`)

| Field | Type | Role | Purpose |
|---|---|---|---|
| `data.scene` | `str` | metadata | Scene id (e.g. `"129_Spot_7"`); used for scene-grouped splits if needed. |
| `data.episode_key` | `str` | metadata | HDF5 group key (== original episode name, e.g. `"129_Spot_7_115"`). |
| `data.real` | `int` | metadata | `1` for real data, `0` for synthetic. Always `1` here (human_oracle is fully real). |
| `data.num_humans` | `int` | metadata | `P`, for convenience and sanity checks. |
| `data.num_timesteps` | `int` | metadata | `T = 10` for SocialNav-SUB. |

## Batching behavior (PyG `DataLoader`)

When wrapped with `torch_geometric.loader.DataLoader`, PyG concatenates per-type
node tensors and offsets edge indices automatically. Verified with `batch_size=4`:

```
HeteroDataBatch(
  scene=[4],                        # list, 4 strings
  episode_key=[4],
  real=[4],
  num_humans=[4],
  num_timesteps=[4],
  robot={ x=[4, 30], traj=[4, 10, 3], batch=[4], ptr=[5] },
  human={ x=[29, 30], traj=[29, 10, 3], y=[29], batch=[29], ptr=[5] },
  (robot, to, human)={  edge_index=[2, 29],   edge_attr=[29, 10, 8] },
  (human, to, robot)={  edge_index=[2, 29],   edge_attr=[29, 10, 8] },
  (human, to, human)={  edge_index=[2, 240],  edge_attr=[240, 10, 8] }
)
```

- `data['robot'].batch[i]` ∈ `{0..B-1}` indicates which graph robot `i` came from.
- `data['human'].batch[j]` ∈ `{0..B-1}` indicates which graph human `j` came from.
- Edge indices are offset internally so `(robot,to,human).edge_index` indexes
  into the concatenated robot/human tensors.

## Sanity checks performed

- Total edges in each graph = `(1+P)*P` (verified across small/medium/large samples).
- `data['human'].y.shape[0] == P` for all graphs.
- Label distribution across all graphs (372 labels total) exactly matches
  `human_oracle/generation_summary.json`:
  `not_consider=294, avoid=43, follow=29, yield=6, overtake=0`.
- Coordinate frame check: every robot trajectory's t=0 row is `[0, 0, 0]`
  (robot starts at origin in its own frame).
- Goal coverage: 60/60 graphs have `data['robot'].goal` set; PyG batches
  stack them into `[B, 2]` correctly.

## Not yet wired in (future)

- Synthetic episodes — none in `human_oracle/` (`real=1` for all 60).
- LLM label sources (`gemini`, `gpt4o`, `o4-mini`) — explicitly excluded
  (Strategy A: human-annotated ground truth only).
- Action probabilities — the source JSONs include `people_action_probabilities`
  (per-human distribution over 5 classes). Currently we expose only the argmax
  via `data['human'].y`. Could be wired through if we want soft-target training.
