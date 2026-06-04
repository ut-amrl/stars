import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

DATASET_SEAN       = 0
DATASET_SSUB_REAL  = 1
DATASET_SSUB_SYNTH = 2
DATASET_SWM        = 3

NODE_FEATURE_DIM = 592


class UnifiedGraphDataset(Dataset):

    def __init__(self, h5_path, ssub=True, robot_mean=None, robot_std=None):
        super().__init__()
        self.h5_path    = h5_path
        self.ssub       = ssub
        self.robot_mean = robot_mean
        self.robot_std  = robot_std

        with h5py.File(h5_path, "r") as f:
            if ssub:
                self._keys = []
                f.visititems(
                    lambda name, obj: self._keys.append(name)
                    if isinstance(obj, h5py.Group) and self._is_sample_group(obj)
                    else None
                )
                self._keys = sorted(self._keys)
                self._scenes     = [str(f[k].attrs.get("scene", "")) for k in self._keys]
                self._dataset_ids = [
                    DATASET_SSUB_REAL if int(f[k].attrs.get("real", 1)) == 1
                    else DATASET_SSUB_SYNTH
                    for k in self._keys
                ]
            else:
                self._keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
                self._scenes     = [str(f[k].attrs["scene"])   for k in self._keys]
                self._dataset_ids = [int(f[k].attrs["dataset"]) for k in self._keys]

    @staticmethod
    def _is_sample_group(grp):
        required = {"node_type", "edge_index", "edge_attr", "action_label"}
        return required.issubset(grp.keys())

    def len(self):
        return len(self._keys)

    def __len__(self):
        return len(self._keys)

    @property
    def scenes(self):
        return self._scenes

    @property
    def dataset_ids(self):
        return self._dataset_ids

    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        key = self._keys[idx]

        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            node_type    = grp["node_type"][:]
            edge_index   = grp["edge_index"][:]
            edge_attr    = grp["edge_attr"][:]
            action_label = grp["action_label"][:]
            scene        = str(grp.attrs.get("scene", ""))

            action_probs = (torch.from_numpy(grp["action_probs"][:])
                           if "action_probs" in grp else None)

            if self.ssub:
                dataset_id = (DATASET_SSUB_REAL
                              if int(grp.attrs.get("real", 1)) == 1
                              else DATASET_SSUB_SYNTH)
            else:
                dataset_id    = int(grp.attrs["dataset"])
                node_features = grp["node_features"][:]
                valid_mask    = grp["valid_mask"][:]
                emotion_label = grp["emotion_label"][:]

        if not self.ssub and self.robot_mean is not None and self.robot_std is not None:
            if np.abs(node_features[0]).sum() > 0:
                node_features = node_features.copy()
                node_features[0] = (
                    (node_features[0] - self.robot_mean.numpy())
                    / np.maximum(self.robot_std.numpy(), 1.0)
                )

        data = Data(
            node_type    = torch.from_numpy(node_type),
            edge_index   = torch.from_numpy(edge_index),
            edge_attr    = torch.from_numpy(edge_attr),
            action_label = torch.from_numpy(action_label),
            dataset_id   = torch.tensor([dataset_id], dtype=torch.long),
        )

        if not self.ssub:
            data.x             = torch.from_numpy(node_features)
            data.valid_mask    = torch.from_numpy(valid_mask)
            data.emotion_label = torch.from_numpy(emotion_label)

        if action_probs is not None:
            data.action_probs = action_probs

        data.scene = scene
        return data

    def set_robot_stats(self, train_indices):
        if self.ssub:
            print("set_robot_stats: skipped (SSUB-only format has no robot features)")
            return

        robot_feats = []
        with h5py.File(self.h5_path, "r") as f:
            for idx in train_indices:
                key  = self._keys[idx]
                feat = f[key]["node_features"][0]
                if np.abs(feat).sum() > 0:
                    robot_feats.append(feat)

        if not robot_feats:
            print("set_robot_stats: no non-zero robot features found in training split")
            return

        robot_feats = np.stack(robot_feats, axis=0)
        self.robot_mean = torch.from_numpy(robot_feats.mean(axis=0).astype(np.float32))
        self.robot_std  = torch.from_numpy(robot_feats.std(axis=0).astype(np.float32))
        tiny_std = self.robot_std < 1e-6
        if tiny_std.any():
            self.robot_std = self.robot_std.clone()
            self.robot_std[tiny_std] = 1.0
        print(
            f"Robot feature stats  n={len(robot_feats)} SEAN samples "
            f"(of {len(train_indices)} total train): "
            f"mean={self.robot_mean.abs().mean():.4f}  "
            f"std={self.robot_std.mean():.4f}  "
            f"clamped_std_dims={int(tiny_std.sum())}"
        )

    def scene_split(self, val_ratio, test_ratio, seed=42,
                    ssub_val_ratio=0.15, ssub_test_ratio=0.0):
        rng = np.random.default_rng(seed)
        names = {DATASET_SEAN: "SEAN", DATASET_SSUB_REAL: "SSUB_real",
                 DATASET_SSUB_SYNTH: "SSUB_synth", DATASET_SWM: "SWM"}

        source_scenes = {}
        for i, (scene, did) in enumerate(zip(self._scenes, self._dataset_ids)):
            source_scenes.setdefault(did, []).append((i, scene))

        test_scenes_set, val_scenes_set = set(), set()
        for did, items in sorted(source_scenes.items()):
            unique = list(dict.fromkeys(sc for _, sc in items))
            unique = rng.permutation(unique).tolist()
            n      = len(unique)
            if did in (DATASET_SEAN, DATASET_SWM):
                n_test = max(1, round(test_ratio * n))
                n_val  = max(1, round(val_ratio  * n))
            else:
                n_test = max(1, round(ssub_test_ratio * n)) if ssub_test_ratio > 0 else 0
                n_val  = max(1, round(ssub_val_ratio   * n))
            test_scenes_set.update(unique[:n_test])
            val_scenes_set.update(unique[n_test: n_test + n_val])
            print(f"  {names.get(did, did)}: {n} scenes → "
                  f"test={n_test}, val={n_val}, train={n - n_test - n_val}")

        train_idx, val_idx, test_idx = [], [], []
        for i, scene in enumerate(self._scenes):
            if scene in test_scenes_set:
                test_idx.append(i)
            elif scene in val_scenes_set:
                val_idx.append(i)
            else:
                train_idx.append(i)

        print(
            f"Scene split  (seed={seed})  "
            f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
        )
        return train_idx, val_idx, test_idx

    def filter_by_source(self, indices, source):
        return [i for i in indices if self._dataset_ids[i] == source]

    def source_counts(self, train_idx, val_idx, test_idx):
        names = {DATASET_SEAN: "SEAN", DATASET_SSUB_REAL: "SSUB_real",
                 DATASET_SSUB_SYNTH: "SSUB_synth", DATASET_SWM: "SWM"}
        for split_name, split in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            counts = {}
            for i in split:
                src = self._dataset_ids[i]
                counts[src] = counts.get(src, 0) + 1
            parts = "  ".join(f"{names[s]}={counts.get(s,0)}" for s in sorted(names))
            print(f"  {split_name:5s}: {len(split):5d}  [{parts}]")
