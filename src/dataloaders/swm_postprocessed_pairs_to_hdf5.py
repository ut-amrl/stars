#!/usr/bin/env python3
"""
Convert SWM postprocessed bag outputs directly into forecasting HDF5.

Expected inputs under --input-root (recursively):
  *_postprocessed_raw.json
  *_postprocessed_interp_kalman.json

For each episode frame, one HDF5 group is written at:
  <episode_key>/segments/segment_<index>

Each group stores:
  - segment_payload_json (required)
  - raw_episode_json / interp_kalman_episode_json (optional)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

try:
    import h5py  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    h5py = None


RAW_SUFFIX = "_postprocessed_raw.json"
INTERP_SUFFIX = "_postprocessed_interp_kalman.json"


def _next_available_group_key(hf: Any, base_key: str) -> str:
    if base_key not in hf:
        return base_key

    suffix = 1
    while True:
        candidate = f"{base_key}_{suffix:03d}"
        if candidate not in hf:
            return candidate
        suffix += 1


def _episode_key_from_path(input_root: Path, raw_file: Path) -> str:
    rel_parent = raw_file.parent.relative_to(input_root)
    if str(rel_parent) in {"", "."}:
        return raw_file.parent.name or "default_episode"
    return str(rel_parent).replace("\\", "/")


def _discover_pairs(input_root: Path) -> tuple[list[tuple[Path, Path, str]], list[Path]]:
    raw_files = sorted([path for path in input_root.rglob(f"*{RAW_SUFFIX}") if path.is_file()])

    pairs: list[tuple[Path, Path, str]] = []
    missing_interp: list[Path] = []
    for raw_file in raw_files:
        interp_name = raw_file.name.replace(RAW_SUFFIX, INTERP_SUFFIX)
        interp_file = raw_file.with_name(interp_name)
        if not interp_file.is_file():
            missing_interp.append(raw_file)
            continue

        episode_key = _episode_key_from_path(input_root, raw_file)
        pairs.append((raw_file, interp_file, episode_key))

    return pairs, missing_interp


def _frame_idx(episode: dict[str, Any], fallback: int) -> int:
    value = episode.get("frame_idx", fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _episodes_by_frame(episodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            continue
        out[_frame_idx(episode, idx)] = episode
    return out


def convert_postprocessed_pairs_to_hdf5(
    *,
    input_root: Path,
    output_path: Path,
    overwrite: bool,
    include_legacy_episode_datasets: bool,
    allow_missing_pairs: bool,
) -> tuple[int, int, int]:
    if h5py is None:
        raise ImportError("h5py is required to create SWM forecasting HDF5 datasets.")

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root directory not found: {input_root}")

    pairs, missing_interp = _discover_pairs(input_root)
    if missing_interp and not allow_missing_pairs:
        missing_example = "\n".join(str(path) for path in missing_interp[:5])
        raise RuntimeError(
            "Found raw postprocessed files without matching interp files. "
            "Use --allow-missing-pairs to skip them.\n"
            f"Examples:\n{missing_example}"
        )

    if not pairs:
        raise RuntimeError(f"No raw+interp postprocessed pairs found under: {input_root}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = h5py.string_dtype(encoding="utf-8")

    written_groups = 0
    duplicate_keys = 0

    with h5py.File(output_path, "w") as hf:
        hf.attrs["format"] = "swm_segments_forecasting_v1"
        hf.attrs["source"] = "swm_postprocessed_pairs"
        hf.attrs["input_root"] = str(input_root)
        hf.attrs["pair_count"] = int(len(pairs))

        for pair_idx, (raw_file, interp_file, episode_key) in enumerate(pairs, start=1):
            print(f"[{pair_idx}/{len(pairs)}] Loading pair: {raw_file.name}")

            raw_payload = json.loads(raw_file.read_text())
            interp_payload = json.loads(interp_file.read_text())

            raw_episodes = raw_payload.get("episodes", []) if isinstance(raw_payload, dict) else []
            interp_episodes = interp_payload.get("episodes", []) if isinstance(interp_payload, dict) else []
            if not isinstance(raw_episodes, list):
                raw_episodes = []
            if not isinstance(interp_episodes, list):
                interp_episodes = []

            raw_by_frame = _episodes_by_frame(raw_episodes)
            interp_by_frame = _episodes_by_frame(interp_episodes)
            all_frames = sorted(set(raw_by_frame.keys()) | set(interp_by_frame.keys()))

            if not all_frames:
                print(f"  Skipping (no episode frames): {raw_file}")
                del raw_payload, interp_payload, raw_episodes, interp_episodes, raw_by_frame, interp_by_frame
                gc.collect()
                continue

            bag_written = 0
            for segment_index, frame_idx in enumerate(all_frames, start=1):
                raw_episode = raw_by_frame.get(frame_idx)
                interp_episode = interp_by_frame.get(frame_idx)

                payload: dict[str, Any] = {}
                if isinstance(raw_episode, dict):
                    payload["raw_episode"] = raw_episode
                if isinstance(interp_episode, dict):
                    payload["interp_kalman_episode"] = interp_episode
                if not payload:
                    continue

                if episode_key and episode_key != ".":
                    base_group_key = f"{episode_key}/segments/segment_{segment_index:06d}"
                else:
                    base_group_key = f"segments/segment_{segment_index:06d}"

                group_key = _next_available_group_key(hf, base_group_key)
                if group_key != base_group_key:
                    duplicate_keys += 1

                group = hf.create_group(group_key)
                group.attrs["episode_key"] = episode_key or "default_episode"
                group.attrs["segment_index"] = int(segment_index)
                group.attrs["frame_idx"] = int(frame_idx)
                group.attrs["source_raw_json_path"] = str(raw_file)
                group.attrs["source_interp_json_path"] = str(interp_file)
                group.attrs["sample_idx"] = int(written_groups)

                payload_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                group.create_dataset("segment_payload_json", data=payload_text, dtype=dt)

                if include_legacy_episode_datasets:
                    if "raw_episode" in payload:
                        group.create_dataset(
                            "raw_episode_json",
                            data=json.dumps(payload["raw_episode"], separators=(",", ":"), ensure_ascii=False),
                            dtype=dt,
                        )
                    if "interp_kalman_episode" in payload:
                        group.create_dataset(
                            "interp_kalman_episode_json",
                            data=json.dumps(payload["interp_kalman_episode"], separators=(",", ":"), ensure_ascii=False),
                            dtype=dt,
                        )

                written_groups += 1
                bag_written += 1

            print(f"  Wrote groups from pair: {bag_written}")

            del raw_payload, interp_payload, raw_episodes, interp_episodes, raw_by_frame, interp_by_frame, all_frames
            gc.collect()

    return written_groups, duplicate_keys, len(missing_interp)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SWM *_postprocessed_raw/interp_kalman JSON pairs into forecasting HDF5."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Root folder to scan recursively for *_postprocessed_raw.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--include-legacy-episode-datasets",
        action="store_true",
        help="Also store raw_episode_json and interp_kalman_episode_json datasets per group.",
    )
    parser.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Skip raw files that do not yet have matching interp files.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    written, duplicates, missing = convert_postprocessed_pairs_to_hdf5(
        input_root=input_root,
        output_path=output_path,
        overwrite=bool(args.overwrite),
        include_legacy_episode_datasets=bool(args.include_legacy_episode_datasets),
        allow_missing_pairs=bool(args.allow_missing_pairs),
    )

    print(f"Wrote HDF5 groups: {written}")
    print(f"Duplicate group names resolved: {duplicates}")
    print(f"Missing interp pairs skipped: {missing}")
    print(f"Saved HDF5: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
