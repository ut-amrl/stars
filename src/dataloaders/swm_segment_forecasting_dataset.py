"""
Dataset for SWM dense-bag segment forecasting.

Each sample is one segment graph with:
- Node 0: robot
- Node 1..P: pedestrians with valid supervision at target_offset_frame

Model inputs are homogeneous graph tensors compatible with HGNNCombinedAE:
- data.node_type   [N]        (0=robot, 1=human)
- data.edge_index  [2, E]     fully connected directed edges (no self-loops)
- data.edge_attr   [E, Tm, 8] relative SE(2) + temporal deltas, interpolated to model_timesteps
- data.x           [N, 592]   zero features (kept for model compatibility)
- data.valid_mask  [N, Tm]    all ones

Supervision fields:
- data.target_xy    [N, 2]      future position at target_offset_frame
- data.target_valid [N]         valid supervision mask (False for robot)
- data.baseline_xy  [N, 2]      linear baseline endpoint prediction
- data.future_xy    [N, H, 2]   GT future trajectory from clip_end+1 .. target_offset
- data.future_valid [N, H]      valid mask for future_xy
- data.obs_xy       [N, To, 2]  observed history positions in frame-offset timebase
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import h5py  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised only when h5py is unavailable.
    h5py = None
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset


NODE_TYPE_ROBOT = 0
NODE_TYPE_HUMAN = 1
EDGE_FEATURE_DIM = 8
ROBOT_FEATURE_DIM = 592
H5_SEGMENT_PAYLOAD_KEY = "segment_payload_json"


def _decode_h5_scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_h5_scalar_text(value.item())
        if value.size == 1:
            return _decode_h5_scalar_text(value.reshape(-1)[0])
        if value.dtype == np.uint8:
            return bytes(value.tolist()).decode("utf-8")

    if hasattr(value, "decode"):
        return str(value.decode("utf-8"))
    return str(value)


def _is_h5_segment_group(group: Any) -> bool:
    keys = set(group.keys())
    if H5_SEGMENT_PAYLOAD_KEY in keys:
        return True
    legacy_keys = {
        "raw_episode_json",
        "interp_kalman_episode_json",
        "raw_episode",
        "interp_kalman_episode",
    }
    return any(key in keys for key in legacy_keys)


def _se2_relative_pose(pos_i: np.ndarray, ori_i: np.ndarray, pos_j: np.ndarray, ori_j: np.ndarray) -> np.ndarray:
    cos_i, sin_i = float(ori_i[0]), float(ori_i[1])
    dx = float(pos_j[0]) - float(pos_i[0])
    dy = float(pos_j[1]) - float(pos_i[1])
    rel_x = cos_i * dx + sin_i * dy
    rel_y = -sin_i * dx + cos_i * dy

    cos_j, sin_j = float(ori_j[0]), float(ori_j[1])
    cos_rel = cos_j * cos_i + sin_j * sin_i
    sin_rel = sin_j * cos_i - cos_j * sin_i
    return np.array([rel_x, rel_y, cos_rel, sin_rel], dtype=np.float32)


def _interpolate_edge_attr(edge_attr: np.ndarray, target_size: int) -> np.ndarray:
    e_count, time_count, feat_dim = edge_attr.shape
    if e_count == 0:
        return np.zeros((0, target_size, feat_dim), dtype=np.float32)
    if time_count == target_size:
        return edge_attr.astype(np.float32)

    tensor = torch.from_numpy(edge_attr.transpose(0, 2, 1).reshape(1, e_count * feat_dim, time_count))
    tensor = F.interpolate(tensor, size=target_size, mode="linear", align_corners=True)
    return tensor.numpy().reshape(e_count, feat_dim, target_size).transpose(0, 2, 1).astype(np.float32)


def _dedupe_sorted_times_xy(times: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    x_vals = x_vals[order]
    y_vals = y_vals[order]

    merged: dict[int, tuple[float, float]] = {}
    for t_val, x_val, y_val in zip(times.tolist(), x_vals.tolist(), y_vals.tolist()):
        merged[int(t_val)] = (float(x_val), float(y_val))

    t_out = np.array(sorted(merged.keys()), dtype=np.float32)
    x_out = np.array([merged[int(t)][0] for t in t_out], dtype=np.float32)
    y_out = np.array([merged[int(t)][1] for t in t_out], dtype=np.float32)
    return t_out, x_out, y_out


def _xy_to_theta(xy: np.ndarray) -> np.ndarray:
    theta = np.zeros((xy.shape[0],), dtype=np.float32)
    last_theta = 0.0
    for idx in range(1, xy.shape[0]):
        dx = float(xy[idx, 0] - xy[idx - 1, 0])
        dy = float(xy[idx, 1] - xy[idx - 1, 1])
        if abs(dx) + abs(dy) > 1e-8:
            last_theta = float(np.arctan2(dy, dx))
        theta[idx] = last_theta
    return theta


def _linear_endpoint_baseline(times: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray, target_time: float) -> tuple[float, float]:
    if times.shape[0] < 2:
        return float(x_vals[-1]), float(y_vals[-1])

    try:
        coeff_x = np.polyfit(times, x_vals, deg=1)
        coeff_y = np.polyfit(times, y_vals, deg=1)
        pred_x = float(coeff_x[0] * target_time + coeff_x[1])
        pred_y = float(coeff_y[0] * target_time + coeff_y[1])
        return pred_x, pred_y
    except Exception:
        return float(x_vals[-1]), float(y_vals[-1])


def _parse_prefixed_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return None


def _extract_segment_index_from_path(file_path: Path) -> int:
    segment_index = _parse_prefixed_index(file_path.parent.name, "segment_")
    if segment_index is None:
        raise ValueError(f"Expected segment_* parent folder for file: {file_path}")
    return int(segment_index)


def _infer_episode_key_from_path(file_path: Path, segments_root: Path) -> str:
    rel = file_path.relative_to(segments_root)
    prefix_parts = list(rel.parts[:-2])

    # Common layout is <episode>/segments/segment_x/segment_x.json.
    if prefix_parts and prefix_parts[-1] == "segments":
        prefix_parts = prefix_parts[:-1]

    if not prefix_parts:
        return "default_episode"
    return "/".join(prefix_parts)


def _segment_file_sort_key(file_path: Path, segments_root: Path) -> tuple[str, int, str]:
    return (
        _infer_episode_key_from_path(file_path, segments_root),
        _extract_segment_index_from_path(file_path),
        str(file_path),
    )


def _infer_episode_key_from_h5_group_key(group_key: str) -> str:
    parts = [part for part in group_key.split("/") if part]
    if not parts:
        return "default_episode"

    if _parse_prefixed_index(parts[-1], "segment_") is not None:
        parts = parts[:-1]
    if parts and parts[-1] == "segments":
        parts = parts[:-1]
    if not parts:
        return "default_episode"
    return "/".join(parts)


def _read_preprocessed_h5_chunk(args):
    h5_path, indices = args
    import h5py
    results = []
    with h5py.File(h5_path, "r") as f:
        for idx in indices:
            grp = f[f"sample_{idx}"]
            results.append({
                "sample_idx": idx,
                "h5_group_key": f"sample_{idx}",
                "episode_key": grp.attrs.get("src_episode_key", grp.attrs.get("episode_key", "default_episode")),
                "segment_index": int(grp.attrs.get("src_segment_index", grp.attrs.get("segment_index", 0))),
                "frame_idx": int(grp.attrs.get("src_frame_idx", grp.attrs.get("frame_idx", -1))),
            })
    return results


def _extract_segment_index_from_h5_group_key(group_key: str) -> int | None:
    parts = [part for part in group_key.split("/") if part]
    if not parts:
        return None
    return _parse_prefixed_index(parts[-1], "segment_")


class SWMSegmentForecastingDataset(Dataset):
    def __init__(
        self,
        segments_root: str | Path,
        *,
        use_smoothed_episode: bool = True,
        clip_end_frame: int = 75,
        target_offset_frame: int = 105,
        history_steps: int = 21,
        model_timesteps: int = 40,
        min_history_points: int = 2,
        require_future_labels: bool = True,
    ):
        super().__init__(root=None)

        self.source_path = Path(segments_root).expanduser()
        self.segments_root = self.source_path
        self.use_smoothed_episode = bool(use_smoothed_episode)
        self.clip_end_frame = int(clip_end_frame)
        self.target_offset_frame = int(target_offset_frame)
        self.history_steps = int(history_steps)
        self.model_timesteps = int(model_timesteps)
        self.min_history_points = int(min_history_points)
        self.require_future_labels = bool(require_future_labels)

        if self.target_offset_frame <= self.clip_end_frame:
            raise ValueError("target_offset_frame must be larger than clip_end_frame")

        self.history_offsets = np.rint(
            np.linspace(0.0, float(self.clip_end_frame), num=self.history_steps)
        ).astype(np.int64)
        self.future_times = np.arange(
            self.clip_end_frame + 1,
            self.target_offset_frame + 1,
            dtype=np.int64,
        )

        self.records: list[dict[str, object]] = []
        if self.source_path.suffix.lower() in {".h5", ".hdf5"}:
            if h5py is None:
                raise ImportError(
                    "h5py is required to load SWM forecasting datasets from HDF5 files."
                )
            if not self.source_path.is_file():
                raise FileNotFoundError(f"HDF5 file not found: {self.source_path}")

            # Check if it is a preprocessed dataset
            with h5py.File(self.source_path, "r") as f:
                first_key = next(iter(f.keys())) if len(f.keys()) > 0 else None
                if first_key and "edge_index" in f[first_key]:
                    self.source_kind = "preprocessed_h5"
                else:
                    self.source_kind = "h5"

            if self.source_kind == "preprocessed_h5":
                source_count, total_tracks, kept_tracks = self._collect_records_from_preprocessed_h5(self.source_path)
                source_label = "samples"
            else:
                source_count, total_tracks, kept_tracks = self._collect_records_from_h5(self.source_path)
                source_label = "groups"
        elif self.source_path.is_dir():
            self.source_kind = "json"
            source_count, total_tracks, kept_tracks = self._collect_records_from_json_dir(self.source_path)
            source_label = "files"
        else:
            raise FileNotFoundError(
                f"Input path does not exist or is unsupported: {self.source_path}. "
                "Expected a segments directory or an .h5/.hdf5 dataset file."
            )

        if not self.records:
            raise RuntimeError(
                f"No valid forecasting samples found under {self.source_path}. "
                f"Ensure future trajectories reach frame offset {self.target_offset_frame}."
            )

        self.episode_keys = sorted({str(record["episode_key"]) for record in self.records})
        self.episode_to_index = {key: idx for idx, key in enumerate(self.episode_keys)}
        for record in self.records:
            record["episode_index"] = int(self.episode_to_index[str(record["episode_key"])])

        coverage = kept_tracks / max(1, total_tracks)
        print(
            f"[SWMSegmentForecastingDataset] source={self.source_kind} {source_label}={source_count} episodes={len(self.episode_keys)} "
            f"usable_graphs={len(self.records)} "
            f"usable_tracks={kept_tracks}/{total_tracks} ({coverage:.1%}) "
            f"target_t={self.target_offset_frame} "
            f"future_labels={'required' if self.require_future_labels else 'optional'}"
        )

    def _collect_records_from_preprocessed_h5(self, h5_path: Path) -> tuple[int, int, int]:
        from multiprocessing import Pool
        import os

        # Save meta cache inside the same folder as this source file (or next to H5 if writable)
        cache_dir = Path(__file__).parent.resolve()
        cache_path = cache_dir / f"{h5_path.stem}_meta_cache.pt"

        if cache_path.is_file():
            print(f"Loading metadata cache from {cache_path}")
            try:
                cache_data = torch.load(cache_path, map_location="cpu")
                self.records = cache_data["records"]
                return cache_data["source_count"], cache_data["total_tracks"], cache_data["kept_tracks"]
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        with h5py.File(h5_path, "r") as handle:
            N = len(handle.keys())

        print(f"Generating metadata for {N} preprocessed H5 samples...")
        num_workers = min(os.cpu_count() or 4, 32)
        indices = list(range(N))
        chunk_size = (N + num_workers - 1) // num_workers
        chunks = [indices[i:i + chunk_size] for i in range(0, N, chunk_size)]

        with Pool(processes=num_workers) as pool:
            tasks = [(h5_path, chunk) for chunk in chunks]
            chunk_results = pool.map(_read_preprocessed_h5_chunk, tasks)

        self.records = []
        for cr in chunk_results:
            self.records.extend(cr)

        self.records.sort(key=lambda r: r["sample_idx"])

        total_tracks = len(self.records)
        kept_tracks = len(self.records)

        try:
            os.makedirs(cache_path.parent, exist_ok=True)
            torch.save({
                "records": self.records,
                "source_count": N,
                "total_tracks": total_tracks,
                "kept_tracks": kept_tracks,
            }, cache_path)
            print(f"Saved metadata cache -> {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save metadata cache: {e}")

        return N, total_tracks, kept_tracks

    def _select_episode_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        primary_key = "interp_kalman_episode" if self.use_smoothed_episode else "raw_episode"
        fallback_key = "raw_episode" if self.use_smoothed_episode else "interp_kalman_episode"

        if primary_key in payload:
            episode = payload[primary_key]
        elif fallback_key in payload:
            episode = payload[fallback_key]
        else:
            raise KeyError(
                f"Segment payload missing both '{primary_key}' and '{fallback_key}' episodes."
            )

        if not isinstance(episode, dict):
            raise TypeError(f"Episode payload for key '{primary_key}' must be a dict.")
        return episode

    def _collect_valid_track_ids(self, episode: dict[str, Any]) -> tuple[list[str], int]:
        people_map = episode.get("people_trajectories_time_x_y", {}) or {}
        future_map = episode.get("future_people_trajectories_time_x_y", {}) or {}
        if not isinstance(people_map, dict):
            return [], 0
        if not isinstance(future_map, dict):
            future_map = {}

        valid_track_ids: list[str] = []
        total_tracks = 0
        for track_id, obs_points in people_map.items():
            total_tracks += 1
            if len(obs_points) < self.min_history_points:
                continue

            if self.require_future_labels:
                future_points = future_map.get(track_id, [])
                if not self._future_has_required_horizon(future_points):
                    continue

            valid_track_ids.append(str(track_id))

        return valid_track_ids, total_tracks

    @staticmethod
    def _list_h5_segment_group_keys(handle: Any) -> list[str]:
        keys: list[str] = []

        def _walk(group: Any, prefix: str) -> None:
            for key in sorted(group.keys()):
                obj = group[key]
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(obj, h5py.Group):
                    if _is_h5_segment_group(obj):
                        keys.append(path)
                    else:
                        _walk(obj, path)

        _walk(handle, "")
        return sorted(keys)

    @staticmethod
    def _read_segment_payload_from_h5_group(group: Any) -> dict[str, Any]:
        if H5_SEGMENT_PAYLOAD_KEY in group:
            text = _decode_h5_scalar_text(group[H5_SEGMENT_PAYLOAD_KEY][()])
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError("HDF5 segment payload must decode to a JSON object.")
            return payload

        payload: dict[str, Any] = {}
        if "raw_episode_json" in group:
            payload["raw_episode"] = json.loads(_decode_h5_scalar_text(group["raw_episode_json"][()]))
        elif "raw_episode" in group:
            payload["raw_episode"] = json.loads(_decode_h5_scalar_text(group["raw_episode"][()]))

        if "interp_kalman_episode_json" in group:
            payload["interp_kalman_episode"] = json.loads(
                _decode_h5_scalar_text(group["interp_kalman_episode_json"][()])
            )
        elif "interp_kalman_episode" in group:
            payload["interp_kalman_episode"] = json.loads(
                _decode_h5_scalar_text(group["interp_kalman_episode"][()])
            )

        if payload:
            return payload
        raise KeyError(
            "HDF5 segment group is missing a supported payload dataset. "
            f"Expected '{H5_SEGMENT_PAYLOAD_KEY}' or legacy *_episode datasets."
        )

    def _collect_records_from_json_dir(self, segments_root: Path) -> tuple[int, int, int]:
        all_files = sorted(
            [path for path in segments_root.rglob("segment_*/segment_*.json") if path.is_file()],
            key=lambda p: _segment_file_sort_key(p, segments_root),
        )

        total_tracks = 0
        kept_tracks = 0
        for sample_idx, file_path in enumerate(all_files):
            segment_index = _extract_segment_index_from_path(file_path)
            episode_key = _infer_episode_key_from_path(file_path, segments_root)

            payload = json.loads(file_path.read_text())
            episode = self._select_episode_payload(payload)
            valid_track_ids, track_count = self._collect_valid_track_ids(episode)
            total_tracks += track_count

            if valid_track_ids:
                kept_tracks += len(valid_track_ids)
                self.records.append(
                    {
                        "sample_idx": sample_idx,
                        "file_path": str(file_path),
                        "episode_key": episode_key,
                        "segment_index": segment_index,
                        "frame_idx": int(episode.get("frame_idx", -1)),
                        "track_ids": valid_track_ids,
                    }
                )

        return len(all_files), total_tracks, kept_tracks

    def _collect_records_from_h5(self, h5_path: Path) -> tuple[int, int, int]:
        total_tracks = 0
        kept_tracks = 0
        with h5py.File(h5_path, "r") as handle:
            group_keys = self._list_h5_segment_group_keys(handle)

            for sample_idx, group_key in enumerate(group_keys):
                group = handle[group_key]
                payload = self._read_segment_payload_from_h5_group(group)
                episode = self._select_episode_payload(payload)
                valid_track_ids, track_count = self._collect_valid_track_ids(episode)
                total_tracks += track_count

                segment_index = group.attrs.get("segment_index")
                if segment_index is None:
                    parsed = _extract_segment_index_from_h5_group_key(group_key)
                    segment_index = parsed if parsed is not None else sample_idx

                episode_key = group.attrs.get("episode_key")
                if episode_key is None:
                    episode_key = _infer_episode_key_from_h5_group_key(group_key)

                frame_idx = int(episode.get("frame_idx", group.attrs.get("frame_idx", -1)))

                if valid_track_ids:
                    kept_tracks += len(valid_track_ids)
                    self.records.append(
                        {
                            "sample_idx": sample_idx,
                            "h5_group_key": str(group_key),
                            "episode_key": str(episode_key),
                            "segment_index": int(segment_index),
                            "frame_idx": frame_idx,
                            "track_ids": valid_track_ids,
                        }
                    )

        return len(group_keys), total_tracks, kept_tracks

    def _load_segment_payload(self, record: dict[str, object]) -> dict[str, Any]:
        if self.source_kind == "h5":
            if not hasattr(self, "_h5_handle") or self._h5_handle is None:
                self._h5_handle = h5py.File(self.source_path, "r")
            group = self._h5_handle[str(record["h5_group_key"])]
            return self._read_segment_payload_from_h5_group(group)

        file_path = Path(str(record["file_path"]))
        return json.loads(file_path.read_text())

    def len(self) -> int:
        return len(self.records)

    def _future_has_required_horizon(self, future_points: list[list[float]]) -> bool:
        if len(future_points) == 0:
            return False

        arr = np.asarray(future_points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return False

        times = arr[:, 0]
        return float(times.min()) <= float(self.future_times[0]) and float(times.max()) >= float(self.target_offset_frame)

    def _step_to_frame_offset(self, step_idx: int) -> int:
        if 0 <= step_idx < len(self.history_offsets):
            return int(self.history_offsets[step_idx])

        ratio = float(step_idx) / max(1.0, float(self.history_steps - 1))
        return int(round(ratio * float(self.clip_end_frame)))

    def _build_person_trajectories(
        self,
        obs_points: list[list[float]],
        future_points: list[list[float]],
    ) -> dict[str, Any] | None:
        obs_arr = np.asarray(obs_points, dtype=np.float32)
        if obs_arr.ndim != 2 or obs_arr.shape[1] < 3:
            return None

        obs_steps = obs_arr[:, 0].astype(np.int64)
        obs_times = np.asarray([self._step_to_frame_offset(int(s)) for s in obs_steps], dtype=np.float32)
        obs_x = obs_arr[:, 1].astype(np.float32)
        obs_y = obs_arr[:, 2].astype(np.float32)
        obs_times, obs_x, obs_y = _dedupe_sorted_times_xy(obs_times, obs_x, obs_y)
        if obs_times.shape[0] < self.min_history_points:
            return None

        hist_times = self.history_offsets.astype(np.float32)
        hist_x = np.interp(hist_times, obs_times, obs_x, left=obs_x[0], right=obs_x[-1]).astype(np.float32)
        hist_y = np.interp(hist_times, obs_times, obs_y, left=obs_y[0], right=obs_y[-1]).astype(np.float32)
        hist_xy = np.stack([hist_x, hist_y], axis=1)

        future_xy = np.zeros((len(self.future_times), 2), dtype=np.float32)
        future_valid = np.zeros((len(self.future_times),), dtype=np.bool_)
        target_xy = hist_xy[-1].copy()
        baseline_x, baseline_y = _linear_endpoint_baseline(obs_times, obs_x, obs_y, float(self.target_offset_frame))
        baseline_xy = np.array([baseline_x, baseline_y], dtype=np.float32)

        has_future_label = False
        fut_arr = np.asarray(future_points, dtype=np.float32)
        if fut_arr.ndim == 2 and fut_arr.shape[0] > 0 and fut_arr.shape[1] >= 3:
            fut_times = fut_arr[:, 0].astype(np.float32)
            fut_x = fut_arr[:, 1].astype(np.float32)
            fut_y = fut_arr[:, 2].astype(np.float32)
            fut_times, fut_x, fut_y = _dedupe_sorted_times_xy(fut_times, fut_x, fut_y)

            if (
                float(fut_times.min()) <= float(self.future_times[0])
                and float(fut_times.max()) >= float(self.target_offset_frame)
            ):
                query_future_times = self.future_times.astype(np.float32)
                future_x = np.interp(query_future_times, fut_times, fut_x).astype(np.float32)
                future_y = np.interp(query_future_times, fut_times, fut_y).astype(np.float32)
                future_xy = np.stack([future_x, future_y], axis=1)
                future_valid[:] = True

                target_x = float(np.interp(float(self.target_offset_frame), fut_times, fut_x))
                target_y = float(np.interp(float(self.target_offset_frame), fut_times, fut_y))
                target_xy = np.array([target_x, target_y], dtype=np.float32)
                has_future_label = True

        if self.require_future_labels and not has_future_label:
            return None

        return {
            "obs_xy": hist_xy,
            "future_xy": future_xy,
            "future_valid": future_valid,
            "target_xy": target_xy,
            "baseline_xy": baseline_xy,
            "has_future_label": np.bool_(has_future_label),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_h5_handle" in state:
            del state["_h5_handle"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5_handle = None

    def get(self, idx: int) -> Data:
        if self.source_kind == "preprocessed_h5":
            record = self.records[idx]
            if not hasattr(self, "_h5_handle") or self._h5_handle is None:
                self._h5_handle = h5py.File(self.source_path, "r")
            grp = self._h5_handle[str(record["h5_group_key"])]
            node_type = torch.from_numpy(grp["node_type"][:]).long()
            edge_index = torch.from_numpy(grp["edge_index"][:]).long()
            edge_attr = torch.from_numpy(grp["edge_attr"][:]).float()

            x_key = "node_features" if "node_features" in grp else "x"
            x = torch.from_numpy(grp[x_key][:]).float()
            valid_mask = torch.from_numpy(grp["valid_mask"][:]).float()

            target_xy = torch.from_numpy(grp["target_xy"][:]).float()
            target_valid = torch.from_numpy(grp["target_valid"][:]).bool()
            baseline_xy = torch.from_numpy(grp["baseline_xy"][:]).float()

            future_xy = torch.from_numpy(grp["future_xy"][:]).float()
            future_valid = torch.from_numpy(grp["future_valid"][:]).bool()
            obs_xy = torch.from_numpy(grp["obs_xy"][:]).float()

            data = Data(
                x=x,
                node_type=node_type,
                edge_index=edge_index,
                edge_attr=edge_attr,
                valid_mask=valid_mask,
                target_xy=target_xy,
                target_valid=target_valid,
                baseline_xy=baseline_xy,
                future_xy=future_xy,
                future_valid=future_valid,
                obs_xy=obs_xy,
            )
            data.segment_index = int(record["segment_index"])
            data.sample_idx = int(record["sample_idx"])
            data.episode_index = int(record["episode_index"])
            data.frame_idx = int(record["frame_idx"])
            data.episode_key = record["episode_key"]

            return data

        record = self.records[idx]
        payload = self._load_segment_payload(record)
        episode = self._select_episode_payload(payload)

        people_map = episode.get("people_trajectories_time_x_y", {})
        future_map = episode.get("future_people_trajectories_time_x_y", {})

        person_entries: list[dict[str, Any]] = []
        for track_id in record["track_ids"]:
            parsed = self._build_person_trajectories(
                people_map.get(track_id, []),
                future_map.get(track_id, []),
            )
            if parsed is not None:
                person_entries.append(parsed)

        if len(person_entries) == 0:
            sample_ref = str(record.get("h5_group_key", record.get("file_path", "<unknown>")))
            raise RuntimeError(f"Sample {sample_ref} has no valid pedestrians after parsing.")

        n_people = len(person_entries)
        n_nodes = 1 + n_people
        t_hist = len(self.history_offsets)
        h_future = len(self.future_times)

        obs_xy = np.zeros((n_nodes, t_hist, 2), dtype=np.float32)
        future_xy = np.zeros((n_nodes, h_future, 2), dtype=np.float32)
        future_valid = np.zeros((n_nodes, h_future), dtype=np.bool_)
        target_xy = np.zeros((n_nodes, 2), dtype=np.float32)
        baseline_xy = np.zeros((n_nodes, 2), dtype=np.float32)
        target_valid = np.zeros((n_nodes,), dtype=np.bool_)

        # Robot trajectory is placeholder origin from segment generation.
        obs_xy[0, :, :] = 0.0

        for person_idx, entry in enumerate(person_entries, start=1):
            obs_xy[person_idx] = entry["obs_xy"]
            future_xy[person_idx] = entry["future_xy"]
            future_valid[person_idx] = entry["future_valid"]
            target_xy[person_idx] = entry["target_xy"]
            baseline_xy[person_idx] = entry["baseline_xy"]
            target_valid[person_idx] = bool(entry["has_future_label"])

        positions = obs_xy.copy()
        orientations = np.zeros((n_nodes, t_hist, 2), dtype=np.float32)
        orientations[0, :, 0] = 1.0
        orientations[0, :, 1] = 0.0
        for node_idx in range(1, n_nodes):
            theta = _xy_to_theta(positions[node_idx])
            orientations[node_idx, :, 0] = np.cos(theta)
            orientations[node_idx, :, 1] = np.sin(theta)

        pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_index = np.stack(
            [
                np.asarray([src for src, _ in pairs], dtype=np.int64),
                np.asarray([dst for _, dst in pairs], dtype=np.int64),
            ],
            axis=0,
        )

        edge_attr = np.zeros((len(pairs), t_hist, EDGE_FEATURE_DIM), dtype=np.float32)
        for edge_idx, (src_idx, dst_idx) in enumerate(pairs):
            prev = None
            for t_idx in range(t_hist):
                rel = _se2_relative_pose(
                    positions[src_idx, t_idx],
                    orientations[src_idx, t_idx],
                    positions[dst_idx, t_idx],
                    orientations[dst_idx, t_idx],
                )
                if prev is None:
                    d_rel = np.zeros((4,), dtype=np.float32)
                else:
                    d_rel = rel - prev
                edge_attr[edge_idx, t_idx] = np.concatenate([rel, d_rel], axis=0)
                prev = rel

        edge_attr = _interpolate_edge_attr(edge_attr, self.model_timesteps)

        node_type = np.zeros((n_nodes,), dtype=np.int64)
        node_type[1:] = NODE_TYPE_HUMAN

        data = Data(
            x=torch.zeros((n_nodes, ROBOT_FEATURE_DIM), dtype=torch.float32),
            node_type=torch.from_numpy(node_type).long(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            valid_mask=torch.ones((n_nodes, self.model_timesteps), dtype=torch.float32),
            target_xy=torch.from_numpy(target_xy).float(),
            target_valid=torch.from_numpy(target_valid),
            baseline_xy=torch.from_numpy(baseline_xy).float(),
            future_xy=torch.from_numpy(future_xy).float(),
            future_valid=torch.from_numpy(future_valid),
            obs_xy=torch.from_numpy(obs_xy).float(),
        )

        data.segment_index = int(record["segment_index"])
        data.sample_idx = int(record["sample_idx"])
        data.episode_index = int(record["episode_index"])
        data.frame_idx = int(record.get("frame_idx", episode.get("frame_idx", -1)))
        return data
