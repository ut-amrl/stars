#!/usr/bin/env python3
"""
Postprocess dense bag BEV tracks into trajectory episodes.

This script reads a dense per-frame JSON (for example swm_data_dense_bag_200s.json)
and writes two postprocessed JSON files:
1) Sparse trajectories (missing timesteps are kept as gaps).
2) Interpolated + Kalman-smoothed trajectories.

Output episode schema:
{
  "frame_idx": <anchor frame index>,
  "n_waypoints_ahead": <requested horizon waypoints>,
  "effective_waypoints_ahead": <actual waypoints in this episode>,
  "interpolated": <bool>,
  "robot_trajectory": [[t, 0.0, 0.0, 0.0], ...],
  "people_trajectories_time_x_y": {
    "<track_id>": [[t, x, y], ...],
    ...
  },
  "future_people_trajectories_time_x_y": {
    "<track_id>": [[t, x, y], ...],
    ...
  }
}

Notes on sampling:
- Horizon in frames = round(source_fps * horizon_seconds).
- In stride mode (default), sampled offsets are: 0, stride, 2*stride, ...,
  and the horizon endpoint is appended when needed.
- In linspace mode, sampled offsets are rounded from np.linspace over the horizon.
"""

from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


log = logging.getLogger(__name__)


def _is_progress_supported() -> bool:
    return tqdm is not None and sys.stderr.isatty()


def _iter_with_progress(
    iterable: Any,
    *,
    total: int,
    desc: str,
    enabled: bool,
):
    if enabled and _is_progress_supported():
        return tqdm(iterable, total=total, desc=desc, unit="it", dynamic_ncols=True, leave=False)
    return iterable


@dataclass(frozen=True)
class PersonObservation:
    track_id: str
    bev_x: float
    bev_y: float


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_ns: int
    people: tuple[PersonObservation, ...]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _track_id_to_str(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return str(int(text))
    return text


def _round_float(value: float, decimals: int) -> float:
    return float(round(float(value), decimals))


def _normalize_and_sort_points(points: list[list[float]]) -> list[tuple[int, float, float]]:
    by_t: dict[int, tuple[float, float]] = {}
    for row in points:
        if len(row) < 3:
            continue
        timestep = int(row[0])
        by_t[timestep] = (float(row[1]), float(row[2]))
    return [(t, xy[0], xy[1]) for t, xy in sorted(by_t.items(), key=lambda item: item[0])]


def _interpolate_missing_points(points: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    if len(points) <= 1:
        return list(points)

    out: list[tuple[int, float, float]] = [points[0]]
    for prev_pt, next_pt in zip(points, points[1:]):
        t0, x0, y0 = prev_pt
        t1, x1, y1 = next_pt
        dt = t1 - t0
        if dt <= 0:
            continue
        if dt > 1:
            for step in range(1, dt):
                alpha = step / float(dt)
                out.append(
                    (
                        t0 + step,
                        x0 + alpha * (x1 - x0),
                        y0 + alpha * (y1 - y0),
                    )
                )
        out.append(next_pt)
    return out


def _estimate_initial_velocity(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros((2,), dtype=np.float64)
    return (points[-1] - points[0]) / max(points.shape[0] - 1, 1)


def _kalman_smooth_points(
    points: list[tuple[int, float, float]],
    process_noise: float,
    measurement_noise: float,
) -> list[tuple[int, float, float]]:
    if len(points) <= 1:
        return list(points)

    times = np.asarray([point[0] for point in points], dtype=np.int64)
    measurements = np.asarray([[point[1], point[2]] for point in points], dtype=np.float64)

    transition = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    process_cov = float(process_noise) * np.eye(4, dtype=np.float64)
    measurement_cov = float(measurement_noise) * np.eye(2, dtype=np.float64)
    identity = np.eye(4, dtype=np.float64)

    initial_velocity = _estimate_initial_velocity(measurements)
    state = np.array(
        [measurements[0, 0], measurements[0, 1], initial_velocity[0], initial_velocity[1]],
        dtype=np.float64,
    )
    state_cov = np.eye(4, dtype=np.float64)

    forward_states: list[np.ndarray] = []
    forward_covs: list[np.ndarray] = []

    for z in measurements:
        pred_state = transition @ state
        pred_cov = transition @ state_cov @ transition.T + process_cov

        innovation_cov = observation_matrix @ pred_cov @ observation_matrix.T + measurement_cov
        kalman_gain = pred_cov @ observation_matrix.T @ np.linalg.inv(innovation_cov)

        innovation = z - (observation_matrix @ pred_state)
        state = pred_state + kalman_gain @ innovation
        state_cov = (identity - kalman_gain @ observation_matrix) @ pred_cov

        forward_states.append(state.copy())
        forward_covs.append(state_cov.copy())

    smoothed_states = [forward_states[-1]]
    smoothed_covs = [forward_covs[-1]]
    for idx in range(len(forward_states) - 2, -1, -1):
        pred_cov = transition @ forward_covs[idx] @ transition.T + process_cov
        smoother_gain = forward_covs[idx] @ transition.T @ np.linalg.inv(pred_cov)

        smoothed_state = forward_states[idx] + smoother_gain @ (
            smoothed_states[0] - transition @ forward_states[idx]
        )
        smoothed_cov = forward_covs[idx] + smoother_gain @ (smoothed_covs[0] - pred_cov) @ smoother_gain.T

        smoothed_states.insert(0, smoothed_state)
        smoothed_covs.insert(0, smoothed_cov)

    return [(int(t), float(state[0]), float(state[1])) for t, state in zip(times.tolist(), smoothed_states)]


def _process_trajectory_map(
    trajectory_map: dict[str, list[list[float]]],
    *,
    interpolate_missing: bool,
    kalman_smooth: bool,
    process_noise: float,
    measurement_noise: float,
    round_decimals: int,
) -> dict[str, list[list[float]]]:
    processed: dict[str, list[list[float]]] = {}
    for track_id, points in sorted(trajectory_map.items(), key=lambda item: item[0]):
        normalized = _normalize_and_sort_points(points)
        if not normalized:
            continue

        if interpolate_missing:
            normalized = _interpolate_missing_points(normalized)
        if kalman_smooth:
            normalized = _kalman_smooth_points(normalized, process_noise, measurement_noise)

        processed[track_id] = [
            [
                int(timestep),
                _round_float(x, round_decimals),
                _round_float(y, round_decimals),
            ]
            for timestep, x, y in normalized
        ]
    return processed


def _load_dense_bag_frames(input_json: Path, sequence_key: Optional[str]) -> tuple[str, list[FrameObservation]]:
    with open(input_json, "r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)

    if not isinstance(raw_payload, dict):
        raise TypeError(f"Expected top-level JSON object, got {type(raw_payload).__name__}")

    if sequence_key is None:
        if len(raw_payload) != 1:
            raise ValueError(
                "Input JSON has multiple top-level keys. Please pass --sequence-key explicitly. "
                f"Found keys: {sorted(raw_payload.keys())}"
            )
        sequence_key = next(iter(raw_payload.keys()))

    if sequence_key not in raw_payload:
        raise KeyError(f"sequence_key={sequence_key!r} not found in input JSON")

    raw_frames = raw_payload[sequence_key]
    if not isinstance(raw_frames, list):
        raise TypeError(
            f"Expected sequence '{sequence_key}' to be a list, got {type(raw_frames).__name__}"
        )

    frames: list[FrameObservation] = []
    for frame in raw_frames:
        if not isinstance(frame, dict):
            continue

        if "frame_index" not in frame:
            continue
        frame_index = int(frame["frame_index"])
        timestamp_ns = int(frame.get("timestamp_ns", -1))

        people_out: list[PersonObservation] = []
        seen_ids_in_frame: set[str] = set()
        for person in frame.get("people", []):
            if not isinstance(person, dict):
                continue
            if "track_id" not in person or "bev_x" not in person or "bev_y" not in person:
                continue
            track_id = _track_id_to_str(person["track_id"])
            if track_id in seen_ids_in_frame:
                continue
            seen_ids_in_frame.add(track_id)
            people_out.append(
                PersonObservation(
                    track_id=track_id,
                    bev_x=float(person["bev_x"]),
                    bev_y=float(person["bev_y"]),
                )
            )

        frames.append(
            FrameObservation(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                people=tuple(people_out),
            )
        )

    frames.sort(key=lambda item: item.frame_index)
    if not frames:
        raise ValueError("No valid frames with frame_index were found in the selected sequence")
    return sequence_key, frames


def _build_sampling_offsets(
    *,
    source_fps: float,
    horizon_seconds: float,
    sampling_mode: str,
    sample_stride_frames: int,
    n_waypoints_ahead: int,
) -> tuple[list[int], int, int]:
    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    if horizon_seconds <= 0.0:
        raise ValueError("horizon_seconds must be positive")
    if n_waypoints_ahead <= 0:
        raise ValueError("n_waypoints_ahead must be positive")

    horizon_frames = int(round(source_fps * horizon_seconds))
    if horizon_frames <= 0:
        raise ValueError(
            "Computed horizon_frames <= 0. Increase source_fps or horizon_seconds."
        )

    if sampling_mode == "stride":
        if sample_stride_frames <= 0:
            raise ValueError("sample_stride_frames must be positive in stride mode")
        offsets = list(range(0, horizon_frames + 1, sample_stride_frames))
        if offsets[-1] != horizon_frames:
            offsets.append(horizon_frames)
    elif sampling_mode == "linspace":
        total_points = n_waypoints_ahead + 1
        raw_offsets = np.linspace(0.0, float(horizon_frames), num=total_points)
        offsets = []
        for value in np.rint(raw_offsets).astype(np.int64).tolist():
            value = int(value)
            if not offsets or value > offsets[-1]:
                offsets.append(value)
        if offsets[0] != 0:
            offsets.insert(0, 0)
        if offsets[-1] != horizon_frames:
            offsets.append(horizon_frames)
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    effective_waypoints_ahead = len(offsets) - 1
    return offsets, horizon_frames, effective_waypoints_ahead


def _collect_track_observations(
    frames: list[FrameObservation],
) -> tuple[
    dict[int, FrameObservation],
    dict[str, list[tuple[int, float, float]]],
    dict[str, list[int]],
]:
    frame_by_idx: dict[int, FrameObservation] = {frame.frame_index: frame for frame in frames}
    tracks: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    for frame in frames:
        for person in frame.people:
            tracks[person.track_id].append((frame.frame_index, person.bev_x, person.bev_y))

    track_frame_indices: dict[str, list[int]] = {}
    for track_id, obs in tracks.items():
        obs.sort(key=lambda item: item[0])
        track_frame_indices[track_id] = [entry[0] for entry in obs]

    return frame_by_idx, tracks, track_frame_indices


def _sampled_steps_for_anchor(
    *,
    frame_idx: int,
    offsets: list[int],
    frame_index_set: set[int],
    require_full_horizon: bool,
) -> Optional[list[tuple[int, int]]]:
    sampled_steps: list[tuple[int, int]] = []
    for local_step, offset in enumerate(offsets):
        abs_frame_idx = frame_idx + offset
        if abs_frame_idx not in frame_index_set:
            if require_full_horizon:
                return None
            continue
        sampled_steps.append((local_step, abs_frame_idx))
    if not sampled_steps:
        return None
    return sampled_steps


def _collect_sparse_people_trajectories(
    sampled_steps: list[tuple[int, int]],
    frame_by_idx: dict[int, FrameObservation],
) -> dict[str, list[list[float]]]:
    trajectories: dict[str, list[list[float]]] = defaultdict(list)
    for local_step, abs_frame_idx in sampled_steps:
        frame = frame_by_idx[abs_frame_idx]
        for person in frame.people:
            trajectories[person.track_id].append([local_step, person.bev_x, person.bev_y])
    return dict(sorted(trajectories.items(), key=lambda item: item[0]))


def _collect_future_trajectories(
    *,
    start_frame_idx: int,
    track_ids: set[str],
    track_observations: dict[str, list[tuple[int, float, float]]],
    track_frame_indices: dict[str, list[int]],
    max_missing_gap_frames: int,
    future_time_unit: str,
    sample_stride_frames: int,
) -> dict[str, list[list[float]]]:
    if max_missing_gap_frames < 0:
        raise ValueError("max_missing_gap_frames must be >= 0")

    future: dict[str, list[list[float]]] = {}
    for track_id in sorted(track_ids):
        obs = track_observations.get(track_id)
        frames_for_track = track_frame_indices.get(track_id)
        if not obs or not frames_for_track:
            continue

        start_pos = bisect.bisect_left(frames_for_track, start_frame_idx)
        if start_pos >= len(obs):
            continue

        points: list[list[float]] = []
        prev_seen_frame: Optional[int] = None
        for obs_idx in range(start_pos, len(obs)):
            frame_idx, bev_x, bev_y = obs[obs_idx]

            if prev_seen_frame is not None:
                missing_count = frame_idx - prev_seen_frame - 1
                if missing_count >= max_missing_gap_frames:
                    break

            rel_frame = frame_idx - start_frame_idx
            if rel_frame < 0:
                prev_seen_frame = frame_idx
                continue

            if future_time_unit == "sample_step":
                if sample_stride_frames <= 0:
                    raise ValueError("sample_stride_frames must be positive when future_time_unit=sample_step")
                if rel_frame % sample_stride_frames != 0:
                    prev_seen_frame = frame_idx
                    continue
                timestep = rel_frame // sample_stride_frames
            elif future_time_unit == "raw_frame":
                timestep = rel_frame
            else:
                raise ValueError(f"Unsupported future_time_unit: {future_time_unit}")

            points.append([int(timestep), float(bev_x), float(bev_y)])
            prev_seen_frame = frame_idx

        if points:
            future[track_id] = points

    return future


def _build_robot_trajectory(sampled_steps: list[tuple[int, int]]) -> list[list[float]]:
    return [[local_step, 0.0, 0.0, 0.0] for local_step, _ in sampled_steps]


def _build_episode_records(
    *,
    frames: list[FrameObservation],
    offsets: list[int],
    n_waypoints_ahead: int,
    anchor_stride: int,
    max_anchors: Optional[int],
    require_full_horizon: bool,
    max_missing_gap_frames: int,
    future_time_unit: str,
    sample_stride_frames: int,
    show_progress: bool,
    progress_desc: str,
) -> list[dict[str, Any]]:
    if anchor_stride <= 0:
        raise ValueError("anchor_stride must be positive")

    frame_by_idx, track_obs, track_frame_indices = _collect_track_observations(frames)
    frame_index_set = set(frame_by_idx.keys())

    anchor_frames = frames[::anchor_stride]
    records: list[dict[str, Any]] = []

    anchor_iter = _iter_with_progress(
        anchor_frames,
        total=len(anchor_frames),
        desc=progress_desc,
        enabled=show_progress,
    )
    for anchor in anchor_iter:
        if max_anchors is not None and len(records) >= max_anchors:
            break

        sampled_steps = _sampled_steps_for_anchor(
            frame_idx=anchor.frame_index,
            offsets=offsets,
            frame_index_set=frame_index_set,
            require_full_horizon=require_full_horizon,
        )
        if sampled_steps is None:
            continue

        sparse_people = _collect_sparse_people_trajectories(sampled_steps, frame_by_idx)
        track_ids_for_future = set(sparse_people.keys())
        future_people = _collect_future_trajectories(
            start_frame_idx=anchor.frame_index,
            track_ids=track_ids_for_future,
            track_observations=track_obs,
            track_frame_indices=track_frame_indices,
            max_missing_gap_frames=max_missing_gap_frames,
            future_time_unit=future_time_unit,
            sample_stride_frames=sample_stride_frames,
        )

        record = {
            "frame_idx": int(anchor.frame_index),
            "timestamp_ns": int(anchor.timestamp_ns),
            "n_waypoints_ahead": int(n_waypoints_ahead),
            "effective_waypoints_ahead": int(len(sampled_steps) - 1),
            "interpolated": False,
            "robot_trajectory": _build_robot_trajectory(sampled_steps),
            "people_trajectories_time_x_y": sparse_people,
            "future_people_trajectories_time_x_y": future_people,
        }
        records.append(record)

    return records


def _build_smoothed_records(
    raw_records: list[dict[str, Any]],
    *,
    process_noise: float,
    measurement_noise: float,
    round_decimals: int,
    show_progress: bool,
    progress_desc: str,
) -> list[dict[str, Any]]:
    smoothed_records: list[dict[str, Any]] = []
    record_iter = _iter_with_progress(
        raw_records,
        total=len(raw_records),
        desc=progress_desc,
        enabled=show_progress,
    )
    for record in record_iter:
        smoothed = dict(record)
        smoothed["interpolated"] = True
        smoothed["people_trajectories_time_x_y"] = _process_trajectory_map(
            record.get("people_trajectories_time_x_y", {}),
            interpolate_missing=True,
            kalman_smooth=True,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            round_decimals=round_decimals,
        )
        smoothed["future_people_trajectories_time_x_y"] = _process_trajectory_map(
            record.get("future_people_trajectories_time_x_y", {}),
            interpolate_missing=True,
            kalman_smooth=True,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            round_decimals=round_decimals,
        )
        smoothed_records.append(smoothed)
    return smoothed_records


def _plot_episode(ax, episode: dict[str, Any], title: str) -> None:
    import matplotlib.pyplot as plt

    people_map = episode.get("people_trajectories_time_x_y", {}) or {}
    future_map = episode.get("future_people_trajectories_time_x_y", {}) or {}

    ax.scatter([0.0], [0.0], marker="x", s=80, c="black", label="robot")
    cmap = plt.get_cmap("tab20")

    for idx, (track_id, points) in enumerate(sorted(people_map.items(), key=lambda item: item[0])):
        if len(points) == 0:
            continue
        color = cmap(idx % 20)
        arr = np.asarray(points, dtype=np.float32)
        ax.plot(arr[:, 1], arr[:, 2], "-o", linewidth=1.7, markersize=3.0, color=color)
        ax.text(float(arr[-1, 1]), float(arr[-1, 2]), str(track_id), fontsize=8, color=color)

        if track_id in future_map and len(future_map[track_id]) > 1:
            future_arr = np.asarray(future_map[track_id], dtype=np.float32)
            ax.plot(
                future_arr[:, 1],
                future_arr[:, 2],
                "--",
                linewidth=1.1,
                alpha=0.6,
                color=color,
            )

    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("bev_x")
    ax.set_ylabel("bev_y")


def _render_episode_comparison_frame(
    raw_episode: dict[str, Any],
    smoothed_episode: dict[str, Any],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    _plot_episode(axes[0], raw_episode, "Raw sparse")
    _plot_episode(axes[1], smoothed_episode, "Interpolated + Kalman")

    title = (
        f"frame_idx={raw_episode.get('frame_idx')} | "
        f"tracks={len(raw_episode.get('people_trajectories_time_x_y', {}))}"
    )
    fig.suptitle(title)
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _create_preview_gif(
    raw_records: list[dict[str, Any]],
    smoothed_records: list[dict[str, Any]],
    output_path: Path,
    *,
    preview_samples: int,
    preview_fps: float,
) -> None:
    import imageio.v2 as imageio

    if preview_samples <= 0 or not raw_records:
        return

    n = len(raw_records)
    sample_count = min(preview_samples, n)
    sample_indices = np.linspace(0, n - 1, num=sample_count, dtype=np.int64).tolist()

    frames = []
    for idx in sample_indices:
        frames.append(_render_episode_comparison_frame(raw_records[idx], smoothed_records[idx]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(preview_fps, 0.1))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _is_generated_output_json(path: Path) -> bool:
    name = path.name
    if name.endswith("_raw.json"):
        return True
    if name.endswith("_interp_kalman.json"):
        return True
    if "_postprocessed" in name:
        return True
    return False


def _resolve_input_jsons(input_target: Path) -> tuple[str, list[Path]]:
    if input_target.is_file():
        if input_target.suffix.lower() != ".json":
            raise ValueError(f"Input file must be a JSON file: {input_target}")
        return "single-file", [input_target]

    if not input_target.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_target}")

    # Single-folder mode: direct JSON files in the folder.
    direct = sorted(
        [
            path
            for path in input_target.glob("*.json")
            if path.is_file() and not _is_generated_output_json(path)
        ]
    )
    if direct:
        return "single-folder", direct

    # Parent-folder mode: one level down, each child folder may contain JSON files.
    nested = sorted(
        [
            path
            for path in input_target.glob("*/*.json")
            if path.is_file() and not _is_generated_output_json(path)
        ]
    )
    if nested:
        return "parent-folder", nested

    raise FileNotFoundError(
        f"No input JSON files found under: {input_target}. "
        "Expected either JSON files directly in this folder or in immediate child folders."
    )


def _run_postprocess_for_input(
    input_json: Path,
    *,
    sequence_key: Optional[str],
    output_dir: Optional[Path],
    output_stem: str,
    source_fps: float,
    horizon_seconds: float,
    sampling_mode: str,
    sample_stride_frames: int,
    n_waypoints_ahead: int,
    anchor_stride: int,
    max_anchors: Optional[int],
    require_full_horizon: bool,
    max_missing_gap_frames: int,
    future_time_unit: str,
    kalman_process_noise: float,
    kalman_measurement_noise: float,
    round_decimals: int,
    preview_gif_samples: int,
    preview_gif_fps: float,
    disable_preview_gif: bool,
    show_progress: bool,
) -> tuple[Path, Path, Optional[Path]]:
    resolved_input = input_json.expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {resolved_input}")

    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else resolved_input.parent
    )

    loaded_sequence_key, frames = _load_dense_bag_frames(resolved_input, sequence_key)
    offsets, horizon_frames, effective_waypoints_ahead = _build_sampling_offsets(
        source_fps=float(source_fps),
        horizon_seconds=float(horizon_seconds),
        sampling_mode=str(sampling_mode),
        sample_stride_frames=int(sample_stride_frames),
        n_waypoints_ahead=int(n_waypoints_ahead),
    )

    log.info("Loaded sequence '%s' with %s frames from %s", loaded_sequence_key, len(frames), resolved_input)
    log.info(
        "Sampling math: fps=%.3f, horizon=%.3fs -> horizon_frames=%s, offsets=%s",
        source_fps,
        horizon_seconds,
        horizon_frames,
        offsets,
    )
    if sampling_mode == "stride" and n_waypoints_ahead != effective_waypoints_ahead:
        log.warning(
            "Requested n_waypoints_ahead=%s but stride sampling over this horizon yields %s waypoints ahead.",
            n_waypoints_ahead,
            effective_waypoints_ahead,
        )

    progress_prefix = resolved_input.stem

    raw_records = _build_episode_records(
        frames=frames,
        offsets=offsets,
        n_waypoints_ahead=int(n_waypoints_ahead),
        anchor_stride=int(anchor_stride),
        max_anchors=max_anchors,
        require_full_horizon=bool(require_full_horizon),
        max_missing_gap_frames=int(max_missing_gap_frames),
        future_time_unit=str(future_time_unit),
        sample_stride_frames=int(sample_stride_frames),
        show_progress=bool(show_progress),
        progress_desc=f"{progress_prefix}:anchors",
    )
    log.info("Built %s raw episodes", len(raw_records))

    smoothed_records = _build_smoothed_records(
        raw_records,
        process_noise=float(kalman_process_noise),
        measurement_noise=float(kalman_measurement_noise),
        round_decimals=int(round_decimals),
        show_progress=bool(show_progress),
        progress_desc=f"{progress_prefix}:smooth",
    )

    base_meta = {
        "input_json": str(resolved_input),
        "sequence_key": loaded_sequence_key,
        "source_fps": float(source_fps),
        "horizon_seconds": float(horizon_seconds),
        "horizon_frames": int(horizon_frames),
        "sampling_mode": str(sampling_mode),
        "sample_stride_frames": int(sample_stride_frames),
        "sample_offsets_frames": offsets,
        "requested_n_waypoints_ahead": int(n_waypoints_ahead),
        "effective_n_waypoints_ahead": int(effective_waypoints_ahead),
        "future_time_unit": str(future_time_unit),
        "max_missing_gap_frames": int(max_missing_gap_frames),
    }

    raw_payload = {
        "meta": dict(base_meta),
        "episodes": raw_records,
    }
    smoothed_meta = dict(base_meta)
    smoothed_meta.update(
        {
            "interpolated": True,
            "kalman_smoothed": True,
            "kalman_process_noise": float(kalman_process_noise),
            "kalman_measurement_noise": float(kalman_measurement_noise),
        }
    )
    smoothed_payload = {
        "meta": smoothed_meta,
        "episodes": smoothed_records,
    }

    raw_output = resolved_output_dir / f"{output_stem}_raw.json"
    smoothed_output = resolved_output_dir / f"{output_stem}_interp_kalman.json"
    preview_output = resolved_output_dir / f"{output_stem}_preview.gif"

    _write_json(raw_output, raw_payload)
    _write_json(smoothed_output, smoothed_payload)
    log.info("Wrote raw output: %s", raw_output)
    log.info("Wrote interpolated+kalman output: %s", smoothed_output)

    preview_written: Optional[Path] = None
    if not disable_preview_gif:
        try:
            _create_preview_gif(
                raw_records,
                smoothed_records,
                preview_output,
                preview_samples=int(preview_gif_samples),
                preview_fps=float(preview_gif_fps),
            )
            preview_written = preview_output
            log.info("Wrote preview GIF: %s", preview_output)
        except Exception as exc:
            log.warning("Could not create preview GIF (%s)", exc)

    return raw_output, smoothed_output, preview_written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess dense-bag BEV tracks into sparse and smoothed trajectory episodes."
    )
    parser.add_argument(
        "--input-json",
        required=False,
        default=None,
        type=Path,
        help=(
            "Input JSON file or directory. If omitted, the current working directory is auto-detected as "
            "single-folder or parent-folder mode."
        ),
    )
    parser.add_argument("--sequence-key", default=None)

    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--output-stem", default="swm_dense_bag_postprocessed")

    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--horizon-seconds", type=float, default=2.5)
    parser.add_argument("--sampling-mode", choices=("stride", "linspace"), default="stride")
    parser.add_argument("--sample-stride-frames", type=int, default=6)
    parser.add_argument("--n-waypoints-ahead", type=int, default=20)

    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--require-full-horizon", action="store_true")

    parser.add_argument("--max-missing-gap-frames", type=int, default=10)
    parser.add_argument("--future-time-unit", choices=("raw_frame", "sample_step"), default="raw_frame")

    parser.add_argument("--kalman-process-noise", type=float, default=0.2)
    parser.add_argument("--kalman-measurement-noise", type=float, default=0.6)
    parser.add_argument("--round-decimals", type=int, default=6)

    parser.add_argument("--preview-gif-samples", type=int, default=6)
    parser.add_argument("--preview-gif-fps", type=float, default=1.5)
    parser.add_argument("--disable-preview-gif", action="store_true")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of input JSON files to process in parallel. "
            "Use small values (e.g., 2-4) to avoid memory pressure."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an input when both expected output JSON files already exist.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_single_task(task: dict[str, Any]) -> Path:
    _configure_logging(bool(task.get("verbose", False)))
    input_json = Path(task["input_json"])
    _run_postprocess_for_input(
        input_json=input_json,
        sequence_key=task.get("sequence_key"),
        output_dir=task.get("output_dir"),
        output_stem=str(task["output_stem"]),
        source_fps=float(task["source_fps"]),
        horizon_seconds=float(task["horizon_seconds"]),
        sampling_mode=str(task["sampling_mode"]),
        sample_stride_frames=int(task["sample_stride_frames"]),
        n_waypoints_ahead=int(task["n_waypoints_ahead"]),
        anchor_stride=int(task["anchor_stride"]),
        max_anchors=task.get("max_anchors"),
        require_full_horizon=bool(task["require_full_horizon"]),
        max_missing_gap_frames=int(task["max_missing_gap_frames"]),
        future_time_unit=str(task["future_time_unit"]),
        kalman_process_noise=float(task["kalman_process_noise"]),
        kalman_measurement_noise=float(task["kalman_measurement_noise"]),
        round_decimals=int(task["round_decimals"]),
        preview_gif_samples=int(task["preview_gif_samples"]),
        preview_gif_fps=float(task["preview_gif_fps"]),
        disable_preview_gif=bool(task["disable_preview_gif"]),
        show_progress=bool(task["show_progress"]),
    )
    return input_json


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if int(args.num_workers) < 1:
        parser.error("--num-workers must be >= 1")

    progress_requested = not bool(args.no_progress)
    progress_enabled = progress_requested and _is_progress_supported()
    if progress_requested and not progress_enabled:
        if tqdm is None:
            log.info("Progress bars requested but tqdm is not installed. Install tqdm or use --no-progress.")
        elif not sys.stderr.isatty():
            log.info("Progress bars disabled because stderr is not a TTY.")

    input_target = (
        args.input_json.expanduser().resolve()
        if args.input_json is not None
        else Path.cwd().resolve()
    )

    mode, input_json_files = _resolve_input_jsons(input_target)
    log.info("Input mode: %s | files discovered: %s | target=%s", mode, len(input_json_files), input_target)

    success_count = 0
    failure_count = 0
    multi_file = len(input_json_files) > 1
    default_stem_requested = str(args.output_stem) == "swm_dense_bag_postprocessed"

    tasks: list[dict[str, Any]] = []
    skipped_existing = 0
    for input_json in input_json_files:
        if multi_file and default_stem_requested:
            output_stem = f"{input_json.stem}_postprocessed"
        else:
            output_stem = str(args.output_stem)

        if args.output_dir is not None:
            resolved_output_dir = args.output_dir.expanduser().resolve()
        else:
            resolved_output_dir = input_json.parent

        raw_output_path = resolved_output_dir / f"{output_stem}_raw.json"
        interp_output_path = resolved_output_dir / f"{output_stem}_interp_kalman.json"
        if bool(args.skip_existing) and raw_output_path.is_file() and interp_output_path.is_file():
            skipped_existing += 1
            continue

        tasks.append(
            {
                "input_json": str(input_json),
                "sequence_key": args.sequence_key,
                "output_dir": args.output_dir,
                "output_stem": output_stem,
                "source_fps": float(args.source_fps),
                "horizon_seconds": float(args.horizon_seconds),
                "sampling_mode": str(args.sampling_mode),
                "sample_stride_frames": int(args.sample_stride_frames),
                "n_waypoints_ahead": int(args.n_waypoints_ahead),
                "anchor_stride": int(args.anchor_stride),
                "max_anchors": args.max_anchors,
                "require_full_horizon": bool(args.require_full_horizon),
                "max_missing_gap_frames": int(args.max_missing_gap_frames),
                "future_time_unit": str(args.future_time_unit),
                "kalman_process_noise": float(args.kalman_process_noise),
                "kalman_measurement_noise": float(args.kalman_measurement_noise),
                "round_decimals": int(args.round_decimals),
                "preview_gif_samples": int(args.preview_gif_samples),
                "preview_gif_fps": float(args.preview_gif_fps),
                "disable_preview_gif": bool(args.disable_preview_gif),
                "show_progress": bool(progress_enabled and int(args.num_workers) == 1),
                "verbose": bool(args.verbose),
            }
        )

    if skipped_existing:
        log.info("Skipping %s inputs with existing outputs (--skip-existing)", skipped_existing)

    if not tasks:
        log.info("No pending inputs to process.")
        return 0

    if int(args.num_workers) == 1 or len(tasks) <= 1:
        task_iter = _iter_with_progress(
            tasks,
            total=len(tasks),
            desc="files",
            enabled=progress_enabled,
        )
        for idx, task in enumerate(task_iter, start=1):
            input_json = Path(task["input_json"])
            log.info("[%s/%s] Processing input JSON: %s", idx, len(tasks), input_json)
            try:
                _run_single_task(task)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                log.error("Failed processing %s: %s", input_json, exc)
    else:
        log.info("Parallel mode enabled: workers=%s", int(args.num_workers))
        progress_bar = (
            tqdm(total=len(tasks), desc="files", unit="file", dynamic_ncols=True)
            if progress_enabled and _is_progress_supported()
            else None
        )
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            max_tasks_per_child=1,
        ) as executor:
            future_map = {}
            for task in tasks:
                worker_task = dict(task)
                worker_task["show_progress"] = False
                future = executor.submit(_run_single_task, worker_task)
                future_map[future] = Path(task["input_json"])

            for future in as_completed(future_map):
                input_json = future_map[future]
                try:
                    future.result()
                    success_count += 1
                except Exception as exc:
                    failure_count += 1
                    log.error("Failed processing %s: %s", input_json, exc)
                if progress_bar is not None:
                    progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    log.info("Completed processing | success=%s failure=%s", success_count, failure_count)
    if success_count == 0 or failure_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
