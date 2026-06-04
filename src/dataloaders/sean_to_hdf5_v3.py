"""
Build the v3 HDF5 dataset for RF LOOCV baseline — grouped by participant
so the LOOCV train/test split from the original paper can be replicated exactly. We don't need
to build the graph structure since the random forest is just taking in the raw features. 

HDF5 structure:
  hf[participant_id]/sample_N/
      label             [3]   raw 1-5 Likert scores [competence, surprise, intention]
      feat_map_resnet18 [40*512  = 20480]
      feat_goal         [40*2   = 80]
      feat_follower     [40*4   = 160]
      feat_nearby       [40*16*4 = 2560]
      feat_eye          [flattened]
      feat_lip          [flattened]
      feat_gaze         [flattened]
Usage:
  cd dataloaders/
  python sean_to_hdf5_v3.py
  python sean_to_hdf5_v3.py --sean_root SEAN_TOGETHER_Dataset/SEAN_TOGETHER --output sean_together_v3.h5
"""

import argparse
import numpy as np
import torch
import h5py
from pathlib import Path


def _to_np(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return np.asarray(v)


def parse_sean_to_hdf5_v3(sean_root: str, output_path: str):
    sean_root = Path(sean_root)
    folders   = sorted([p for p in sean_root.iterdir() if p.is_dir()])
    print(f"Found {len(folders)} participant folders in {sean_root}")

    total_samples = 0
    skipped       = 0

    with h5py.File(output_path, 'w') as hf:
        participant_ids = [f.name for f in folders]
        hf.attrs["participant_ids"] = participant_ids

        for folder in folders:
            participant_id = folder.name
            pt_files = sorted(
                folder.glob("temporal_*.pt"),
                key=lambda p: int(p.stem.split('_')[1])
            )

            local_idx = 0
            p_grp = hf.create_group(participant_id)

            for pt_file in pt_files:
                try:
                    x = torch.load(str(pt_file), weights_only=False)
                except Exception as e:
                    print(f"  SKIP {participant_id}/{pt_file.name}: {e}")
                    skipped += 1
                    continue

                follower  = _to_np(x["spatial"]["follower"]).astype(np.float32)      # [40, 4]
                nearby    = _to_np(x["spatial"]["nearby"]).astype(np.float32)         # [40, 16, 4]
                map_r18   = _to_np(x["physical"]["map_resnet18"]).astype(np.float32)  # [40, 512]
                goal      = _to_np(x["spatial"]["goal"]).astype(np.float32)           # [40, 2]
                eye       = _to_np(x["facial"]["eye"]).astype(np.float32)
                lip       = _to_np(x["facial"]["lip"]).astype(np.float32)
                gaze      = _to_np(x["facial"]["gaze"]).astype(np.float32)
                label     = _to_np(x["y"]).astype(np.float32)                         # [3] raw 1-5 Likert

                grp = p_grp.create_group(f"sample_{local_idx}")
                grp.create_dataset("label",             data=label)
                grp.create_dataset("feat_map_resnet18", data=map_r18.flatten(),   compression="gzip")
                grp.create_dataset("feat_goal",         data=goal.flatten())
                grp.create_dataset("feat_follower",     data=follower.flatten())
                grp.create_dataset("feat_nearby",       data=nearby.flatten(),    compression="gzip")
                grp.create_dataset("feat_eye",          data=eye.flatten(),       compression="gzip")
                grp.create_dataset("feat_lip",          data=lip.flatten())
                grp.create_dataset("feat_gaze",         data=gaze.flatten())

                local_idx    += 1
                total_samples += 1

            print(f"  {participant_id}: {local_idx} samples")

    print(f"\nDone. Total samples: {total_samples}  |  Skipped: {skipped}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sean_root", default="SEAN_TOGETHER_Dataset/SEAN_TOGETHER")
    parser.add_argument("--output",    default="sean_together_v3.h5")
    args = parser.parse_args()
    parse_sean_to_hdf5_v3(args.sean_root, args.output)
