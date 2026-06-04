import argparse
import shutil

import h5py


def _sample_indices(handle):
    idxs = []
    for k in handle.keys():
        parts = k.split("_")
        if len(parts) == 2 and parts[0] == "sample" and parts[1].isdigit():
            idxs.append(int(parts[1]))
    return idxs


def merge(base_path, add_path, output_path):
    print(f"[merge] copying base {base_path} -> {output_path}")
    shutil.copyfile(base_path, output_path)

    with h5py.File(output_path, "a") as out, h5py.File(add_path, "r") as add:
        base_idxs = _sample_indices(out)
        next_idx = (max(base_idxs) + 1) if base_idxs else 0
        base_count = len(base_idxs)
        print(f"[merge] base has {base_count} sample_* groups; next index = {next_idx}")

        add_keys = sorted(_sample_indices(add))
        added = 0
        for src_idx in add_keys:
            src_name = f"sample_{src_idx}"
            dst_name = f"sample_{next_idx}"
            add.copy(src_name, out, name=dst_name)
            next_idx += 1
            added += 1

        total = len(_sample_indices(out))
        print(f"[merge] added {added} groups from {add_path}")
        print(f"[merge] total sample_* groups now: {total}  (base {base_count} + add {added})")
    print(f"[merge] saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a unified-format h5 into a copy of a base combined h5.")
    parser.add_argument("--base", default = "combined_v1.h5", help="Base combined h5 (e.g. combined_v1.h5)")
    parser.add_argument("--add", default = "/robodata/speedway/SWM_speedway/cpt_graph_forecasting_10pct_unified.h5", help="Unified h5 to append (e.g. SWM converted file)")
    parser.add_argument("--output", default = "/robodata/social_world_model/combined_v2.h5", help="Output combined h5 (e.g. combined_v2.h5)")
    args = parser.parse_args()

    merge(args.base, args.add, args.output)
