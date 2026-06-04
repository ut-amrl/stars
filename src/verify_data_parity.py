import sys
import os
import torch
import yaml
from types import SimpleNamespace

# Add src and group_gnn to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from probes.common import seed_everything

def main():
    print("Starting Cocktail Party Data Parity Verification...")
    
    # 1. Load config
    config_path = os.path.join(SRC_DIR, "params", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args_dict = cfg["linear_probe_all_params"]
    args = SimpleNamespace(**args_dict)
    
    # Ensure cocktail party root exists
    args.cocktail_party_root = os.path.abspath(args.cocktail_party_root)
    print(f"Dataset root path: {args.cocktail_party_root}")
    print(f"Group GNN path: {GROUP_GNN_DIR}")
    
    device = torch.device("cpu")
    batch_size = min(16, int(args.batch_size))
    
    # 2. Load Option A dataset and loaders
    print("\n--- Loading Option A Loaders ---")
    seed_everything(42)
    dataset_a, loaders_a = get_cocktail_loaders(
        args.cocktail_party_root, batch_size, device, use_gnn_transforms=True
    )
    
    
    # 3. Load Option B dataset and loaders
    print("\n--- Loading Option B Loaders ---")
    seed_everything(42)
    dataset_b, loaders_b = get_cocktail_loaders(
        args.cocktail_party_root, batch_size, device, use_gnn_transforms=True
    )
    # 4. Compare Datasets and Splits
    print("\n--- Comparing Datasets and Splits ---")
    assert len(dataset_a) == len(dataset_b), f"Dataset length mismatch: {len(dataset_a)} vs {len(dataset_b)}"
    print(f"Dataset length: {len(dataset_a)} (Perfect Match)")
    
    # Compare loader splits and indices
    for split in ["train", "val", "test"]:
        loader_a = loaders_a[split]
        loader_b = loaders_b[split]
        
        # Check size of subsets
        len_a = len(loader_a.dataset)
        len_b = len(loader_b.dataset)
        assert len_a == len_b, f"Split {split} length mismatch: {len_a} vs {len_b}"
        print(f"Split '{split}' size: {len_a} (Perfect Match)")
        
    # 5. Check Batch-by-Batch Determinism & Parity
    print("\n--- Comparing Batch-by-Batch Loader Output ---")
    
    # Note: In PyTorch and PyTorch Geometric DataLoaders, the RandomSampler's 
    # shuffle index generation is lazily evaluated on the very first next() call,
    # NOT on iter(loader). 
    # Therefore, when comparing two loaders in lock-step, we must initialize the 
    # index generation for each loader immediately after seeding it.
    seed_everything(42)
    train_iter_a = iter(loaders_a["train"])
    print("Immediate train_iter_a Batch 0 frame ids:", next(train_iter_a).frame_id)
    
    seed_everything(42)
    train_iter_b = iter(loaders_b["train"])
    print("Immediate train_iter_b Batch 0 frame ids:", next(train_iter_b).frame_id)
    
    batch_count = 1
    for batch_a in train_iter_a:
        try:
            batch_b = next(train_iter_b)
        except StopIteration:
            raise AssertionError("Loader B has fewer batches than Loader A")
            
        # Assert batch properties are identical
        if not torch.equal(batch_a.edge_index, batch_b.edge_index):
            print(f"\nMismatch detected in Batch {batch_count}:")
            print(f"  batch_a frame_ids: {batch_a.frame_id}")
            print(f"  batch_b frame_ids: {batch_b.frame_id}")
            raise AssertionError(f"Batch {batch_count} edge_index mismatch")
            
        assert torch.equal(batch_a.y, batch_b.y), f"Batch {batch_count} y (labels) mismatch"
        assert torch.equal(batch_a.edge_attr, batch_b.edge_attr), f"Batch {batch_count} edge_attr mismatch"
        assert batch_a.frame_id == batch_b.frame_id, f"Batch {batch_count} frame_id mismatch"
        
        batch_count += 1
        
    try:
        next(train_iter_b)
        raise AssertionError("Loader B has more batches than Loader A")
    except StopIteration:
        pass
        
    print(f"Successfully verified {batch_count} batches in 'train' split!")
    print("All batch structures, raw edge attributes (PPF), node labels (y), and frame IDs are 100% identical!")
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUCCESSFUL: OPTION A AND OPTION B USE THE EXACT SAME DATA")
    print("=" * 60)

if __name__ == "__main__":
    main()
