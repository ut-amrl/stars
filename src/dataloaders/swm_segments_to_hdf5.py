#!/usr/bin/env python3
"""
Convert SWM forecasting segment JSON files into a single HDF5 dataset.

This mirrors the existing converter -> HDF5 -> loader workflow used by other
pipelines in this repository.

Expected input layout (supports nesting):
  <root>/segment_*/segment_*.json
  <root>/<episode>/segments/segment_*/segment_*.json

Each HDF5 sample group stores:
  - segment_payload_json       full JSON payload text from the segment file
  - raw_episode_json           raw episode JSON text (if present)
  - interp_kalman_episode_json smoothed episode JSON text (if present)

Group attributes:
  - episode_key
  - segment_index
  - frame_idx
  - source_json_path
  - sample_idx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import h5py  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised only when h5py is unavailable.
    h5py = None


def _parse_prefixed_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return None


def _extract_segment_index_from_path(file_path: Path) -> int:
    parent_name = file_path.parent.name
    segment_index = _parse_prefixed_index(parent_name, "segment_")
    if segment_index is None:
        raise ValueError(f"Expected segment_* parent folder for file: {file_path}")
    return int(segment_index)


def _infer_episode_key_from_path(file_path: Path, segments_root: Path) -> str:
    rel = file_path.relative_to(segments_root)
    prefix_parts = list(rel.parts[:-2])

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


def _discover_segment_files(segments_root: Path) -> list[Path]:
    if not segments_root.is_dir():
        raise FileNotFoundError(f"Segments root directory not found: {segments_root}")

    files = [path for path in segments_root.rglob("segment_*/segment_*.json") if path.is_file()]
    return sorted(files, key=lambda p: _segment_file_sort_key(p, segments_root))


def _next_available_group_key(hf: Any, base_key: str) -> str:
    if base_key not in hf:
        return base_key

    suffix = 1
    while True:
        candidate = f"{base_key}_{suffix:03d}"
        if candidate not in hf:
            return candidate
        suffix += 1


def convert_segments_to_hdf5(
    *,
    segments_root: Path,
    output_path: Path,
    overwrite: bool,
) -> tuple[int, int]:
    if h5py is None:
        raise ImportError("h5py is required to create SWM forecasting HDF5 datasets.")

    segment_files = _discover_segment_files(segments_root)
    if not segment_files:
        raise RuntimeError(f"No segment files found under: {segments_root}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dt = h5py.string_dtype(encoding="utf-8")
    written = 0
    duplicates = 0

    with h5py.File(output_path, "w") as hf:
        hf.attrs["format"] = "swm_segments_forecasting_v1"
        hf.attrs["segments_root"] = str(segments_root)

        for sample_idx, file_path in enumerate(segment_files):
            payload = json.loads(file_path.read_text())
            if not isinstance(payload, dict):
                continue

            episode_key = _infer_episode_key_from_path(file_path, segments_root)
            segment_index = _extract_segment_index_from_path(file_path)

            frame_idx = -1
            if isinstance(payload.get("interp_kalman_episode"), dict):
                frame_idx = int(payload["interp_kalman_episode"].get("frame_idx", frame_idx))
            elif isinstance(payload.get("raw_episode"), dict):
                frame_idx = int(payload["raw_episode"].get("frame_idx", frame_idx))

            segment_name = f"segment_{segment_index:06d}"
            if episode_key == "default_episode":
                base_group_key = f"segments/{segment_name}"
            else:
                base_group_key = f"{episode_key}/segments/{segment_name}"

            group_key = _next_available_group_key(hf, base_group_key)
            if group_key != base_group_key:
                duplicates += 1

            grp = hf.create_group(group_key)
            grp.attrs["episode_key"] = episode_key
            grp.attrs["segment_index"] = int(segment_index)
            grp.attrs["frame_idx"] = int(frame_idx)
            grp.attrs["source_json_path"] = str(file_path)
            grp.attrs["sample_idx"] = int(sample_idx)

            payload_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            grp.create_dataset("segment_payload_json", data=payload_text, dtype=dt)

            raw_episode = payload.get("raw_episode")
            if isinstance(raw_episode, dict):
                grp.create_dataset(
                    "raw_episode_json",
                    data=json.dumps(raw_episode, separators=(",", ":"), ensure_ascii=False),
                    dtype=dt,
                )

            interp_episode = payload.get("interp_kalman_episode")
            if isinstance(interp_episode, dict):
                grp.create_dataset(
                    "interp_kalman_episode_json",
                    data=json.dumps(interp_episode, separators=(",", ":"), ensure_ascii=False),
                    dtype=dt,
                )

            written += 1

    return written, duplicates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SWM segment JSON files into a forecasting HDF5 dataset.")
    parser.add_argument(
        "--segments_root",
        default="dataloaders/swm_data_dense_bag_200s/segments",
        type=Path,
        help="Root directory containing segment_*/segment_*.json files.",
    )
    parser.add_argument(
        "--output",
        default="dataloaders/swm_data_dense_bag_200s/swm_segments_forecasting.h5",
        type=Path,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    segments_root = args.segments_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    written, duplicates = convert_segments_to_hdf5(
        segments_root=segments_root,
        output_path=output_path,
        overwrite=bool(args.overwrite),
    )

    print(f"Converted segments: {written}")
    print(f"Duplicate group names resolved: {duplicates}")
    print(f"Saved HDF5: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
