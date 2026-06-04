# STARS: SpatioTemporal Autoencoded Representations for Social-interaction

This repository contains the official implementation of STARS, a SpatioTemporal Autoencoded Representation for Social-interaction. The codebase supports converting pedestrian motion datasets into graph formats, training homogeneous and heterogeneous GNN-VAEs, and running downstream linear probes for social classification tasks.

## Quick Start (Run All Experiments)

Run this command block from the root of the repository to download all required datasets, build the Docker environment, and execute the test harness for all experiments:

```bash
# 1. Download real datasets and pretrained checkpoints
./download_data.sh

# 2. Build the Docker container environment
./container build

# 3. Generate the required heterogeneous human oracle dataset
./container cmd "cd src && python dataloaders/socialnav_to_hdf5_v3.py convert --train_dir ../data/dataset_split/val/ssub --no_synth --output ../data/socialnav_human_oracle_v3.h5"

# 4. Run all experiments and validate results (fast 1-seed sweep)
./container cmd "python src/test_paper_results.py --run-all --fast"
```

*(Omit the `--fast` flag inside the container command if you wish to run the full 10-seed sweeps matching the paper setup.)*

---

## Datasets & Pretrained Models

To run the experiments and test harness using the real datasets and the pretrained heterogeneous model, run the download script from the repository root:

```bash
./download_data.sh
```


### Files to Upload / Share

To release this repository and allow others to download the real datasets and pretrained checkpoints, upload the following files to a public file hosting service (such as **Google Drive**, **Hugging Face**, or **Zenodo**) and update the File IDs/URLs in [download_data.sh](file:///home/ntsoi/all_swm/stars/download_data.sh):

1. **`combined_v1.h5`** (~563MB): Preprocessed unified SEAN+SNSB heterogeneous dataset. Required for model training and probe evaluation.
2. **`social_nav_eval_combined.h5`** (~4.2MB): SNSB evaluation dataset. Required for probe validation.
3. **`split_indices_final.pt`** (~18KB): Final scene splits. Required for splits parity.
4. **`best_model_optuna.pt`** (~16MB): Pretrained heterogeneous GNN-VAE model checkpoint (best model from the Optuna search). Required to run the test harness / probe evaluations without retraining from scratch.
5. **`dataset_splits.zip`** (~500MB): Archive of the 10 data splits (`s1`..`s10`) needed to run the Random Forest baseline sweeps.

**Recommendation:** Upload these files to a public Google Drive folder. You can obtain the direct download file IDs for each file and paste them into the corresponding variables in [download_data.sh](file:///home/ntsoi/all_swm/stars/download_data.sh) (`COMBINED_H5_ID`, `SNS_EVAL_H5_ID`, etc.). Alternatively, host them on **Hugging Face Hub** or **Zenodo** and modify the `curl` commands to fetch from those URLs.

---

## Environment Setup

All dependencies are managed via Docker.

```bash
# Build the container
./container build

# Enter the shell
./container shell
```

Alternatively, you can activate your local python environment and install requirements:
```bash
cd src
# source /path/to/your/venv/bin/activate
```

---

## Running Tests (Reproducibility Validation)

A test harness is provided to verify that the generated metrics match the quantitative results (Table 1 F1-scores) from the paper.

First, navigate to the source directory (or run within the container shell):
```bash
cd src
```

### Run All Experiments and Validate Results
You can run all experiments (both heterogeneous and homogeneous pipelines) and automatically compare the results against the paper's ground truth with a single command:

```bash
# Run a fast, smaller test sweep (1 seed) for all experiments to verify pipeline execution
python test_paper_results.py --run-all --fast

# Run the full test sweep for all experiments (10 seeds, matches the paper setup)
python test_paper_results.py --run-all
```

### Option 2: Run Specific Pipelines
If you only want to test a single pipeline:

```bash
# Run a fast, smaller sweep for the heterogeneous pipeline
python test_paper_results.py --run-hetero --fast

# Run the full heterogeneous pipeline sweep
python test_paper_results.py --run-hetero

# Run a fast, smaller sweep for the homogeneous pipeline
python test_paper_results.py --run-homo --fast

# Run the full homogeneous pipeline sweep
python test_paper_results.py --run-homo
```

### Option 3: Validate Existing Precomputed Results
If you have already run the pipeline and have the output CSV files, you can validate them directly without retraining:

```bash
python test_paper_results.py --csv-sean path/to/summary_sean.csv --csv-sns path/to/summary_snsb.csv
```

---

## Running Experiments

Experiments are divided into three main pipelines: Heterogeneous, Homogeneous, and Downstream Probing. We provide orchestrated shell scripts in `src/hetero_scripts` for reproducibility.

The usual execution order is:
1. `run_train_vae.sh` (Self-supervised pre-training)
2. `run_label_eff.sh` (Downstream linear probe sweeping)
3. `run_summarize.sh` (Results summarization)
4. `visualize_probes_latent.py` (Generate t-SNE and PCA figures for the SNS latent space)

### 1. Heterogeneous Graph VAE Pipeline (STARS-He)

All commands below should be run from the `src/` directory. GPU selection is managed via `CUDA_VISIBLE_DEVICES`. The wrapper scripts schedule at most one training/eval process per GPU at any time.

#### A. Combined (SEAN + SNSB) Model
To train 10 independent unified SEAN+SNSB Heterogeneous VAEs on the fixed data split:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_train_vae.sh \
  --pipeline combined \
  --out-root checkpoints/repro_hetero_optuned128
```

Run the label-efficiency sweep (10 VAEs x 10 runs x 6 fractions):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_hetero_optuned128 \
  --out-prefix label_eff_results_repro_combined
```

Summarize the results:
```bash
./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_combined \
  --merged label_eff_results_repro_combined.csv
```
*Expected merged size: 1201 lines (1 header + 10 VAEs * 6 fractions * 10 runs * 2 tasks).*

#### B. SEAN-Only Model
To train 10 independent SEAN-only heterogeneous VAEs on the fixed data split:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_train_vae.sh \
  --pipeline sean \
  --out-root checkpoints/repro_sean_optuned128
```

Run the label-efficiency sweep (10 VAEs x 10 runs x 6 fractions):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
  --pipeline sean \
  --vae-root checkpoints/repro_sean_optuned128 \
  --out-prefix label_eff_results_repro_sean
```

Summarize the results:
```bash
./hetero_scripts/run_summarize.sh \
  --pipeline sean \
  --in-prefix label_eff_results_repro_sean \
  --merged label_eff_results_repro_sean.csv
```
*Expected merged size: 601 lines (1 header + 10 VAEs * 6 fractions * 10 runs).*

---

### 2. Homogeneous Graph VAE Pipeline (STARS-Ho)

The homogeneous graph model uses config-driven training scripts. To train the homogeneous model:
```bash
python train_hgnn_unified_v2.py --config params/config.yaml
```
Add `--ssub` to run in SSUB-only mode (no `valid_mask`, no node features).

#### Config section to modify

`train_hgnn_unified_v2.py` reads **only** the `homognn_params` section of [`params/config.yaml`](src/params/config.yaml); all other top-level sections (`vae`, `linear_probe`, `optuna_hgnn`, etc.) are ignored by this script. The key fields to set under `homognn_params`:

- **`h5_path`** — the input graph dataset (HDF5) to train on.
- **`experiments_root` / `run_name`** — output location for this run. The script creates `experiments_root/<timestamped run_name>/` and copies the resolved config into it; use a unique `run_name` so checkpoints aren't overwritten.
- **`device`** — the GPU to train on (e.g. `cuda:0`), or `cpu`. One run uses a single GPU.
- **`seed`** — vary this to produce the independent seeds for a multi-seed sweep.
- **Architecture keys** (`hidden_dim`, `latent_dim`, `num_mp_layers`, `edge_transformer_layers`, `num_heads`, `dropout`) — define the encoder. If you later probe this checkpoint, the probe config's matching keys must be identical or weight loading will fail.

---

### 3. Downstream Ablation Studies

All label-efficiency ablations keep the pretrained VAE encoder weights frozen and modify either the probe input or prediction baseline via the `--extra` argument in `run_label_eff.sh`.

#### A. Combined Model Ablations
- **Random encoder**:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
    --pipeline combined --vae-root checkpoints/repro_hetero_optuned128 \
    --out-prefix label_eff_results_repro_random_encoder --extra "--mode random_encoder"
  ```
- **Weighted random baseline**:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
    --pipeline combined --vae-root checkpoints/repro_hetero_optuned128 \
    --out-prefix label_eff_results_repro_wrb --extra "--mode weighted_random_baseline"
  ```
- **Robot-only latent representation**:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
    --pipeline combined --vae-root checkpoints/repro_hetero_optuned128 \
    --out-prefix label_eff_results_repro_robot_only --extra "--ablation robot_only"
  ```
- **Bystander-only latent representation**:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
    --pipeline combined --vae-root checkpoints/repro_hetero_optuned128 \
    --out-prefix label_eff_results_repro_bystander_only --extra "--ablation bystander_only"
  ```

#### B. SEAN-Only Raw Robot Ablation
- **Raw robot 592D features** (probes normalized input features directly):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_label_eff.sh \
    --pipeline sean --vae-root checkpoints/repro_sean_optuned128 \
    --out-prefix label_eff_results_repro_sean_raw_robot --extra "--ablation raw_robot"
  ```

---

### 4. Generating Figures (t-SNE & PCA)

To generate t-SNE and PCA visualization plots for the SNS dataset on the test split:
```bash
CUDA_VISIBLE_DEVICES=0 python visualize_probes_latent.py \
  --checkpoint checkpoints/repro_hetero_optuned128/run0/best_model.pt \
  --split test
```

---

### 5. Hyperparameter Tuning (Optuna)

To run a parallelized Optuna search to tune model hyperparameters:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_optuna.sh \
  --pipeline combined \
  --study repro_unified_256x128 \
  --storage sqlite:///optuna_repro_unified_256x128.db \
  --n-trials 10
```
*Note: `--n-trials` is per-GPU, making the total trials equal to `n_trials * visible_GPUs`.*

---

## Command Line Arguments Reference

### `run_train_vae.sh`
- `--pipeline`: `combined` (SEAN+SNSB) | `sean` (SEAN-only)
- `--n-vaes`: Number of VAE checkpoints to train (default: 10)
- `--out-root`: Directory where checkpoints are saved
- `--hidden-dim` / `--latent-dim`: Dimensions of hidden layers & latent space (default: 256 / 128)
- `--epochs` / `--lr` / `--batch-size` / `--dropout`: Training hyperparameters
- `--extra`: Verbatim string arguments passed to train script

### `run_label_eff.sh`
- `--pipeline`: `combined` | `sean`
- `--vae-root`: Directory containing the pretrained VAE runs
- `--out-prefix`: Prefix for the output CSV results files
- `--no-flip-surprise`: Disables surprise target flipping (SEAN-only pipeline only)
- `--extra`: Parameters passed directly to probe script (e.g. `--mode random_encoder` or `--ablation robot_only`)

### `run_summarize.sh`
- `--pipeline`: `combined` | `sean`
- `--in-prefix`: Prefix of input CSV files to merge
- `--merged`: Filename of the output merged raw CSV file
- `--task`: `sean` | `snsb` | `both`

---

## Output Metrics & Interpretation

`summarize_label_efficiency.py` aggregates probe outputs across runs and computes mean $\pm$ standard deviation for each label fraction:

- **Combined Model (SEAN + SNSB)**:
  - **SEAN**: `f1_macro`, `auc_macro`, plus per-label F1/AUC/accuracy.
  - **SNSB**: `f1_macro3` (mean F1 of follow/avoid/not_consider classes), `f1_macro` (5-class sklearn macro), `acc`, `pa` (prediction accuracy), plus per-class metrics.
- **SEAN-only Model**:
  - `f1_macro`, `auc_macro`, `acc_macro`, plus per-label F1/AUC/accuracy.

---

## Directory Structure & Component Mapping

Below is a detailed map of the core files implementing the heterogeneous GNN-VAE architecture:

- **`src/models/`**:
  - [unified_gnn_vae_v1.py](file:///home/ntsoi/all_swm/stars/src/models/unified_gnn_vae_v1.py): The combined Heterogeneous GNN-VAE (SEAN+SNSB) model. Implements the `HeteroMPLayer` which dynamically refines edge embeddings.
  - [gnn_vae_v4.py](file:///home/ntsoi/all_swm/stars/src/models/gnn_vae_v4.py): The SEAN-only heterogeneous GNN-VAE model.
- **`src/dataloaders/`**:
  - [unified_gnn_vae_datasets_v1.py](file:///home/ntsoi/all_swm/stars/src/dataloaders/unified_gnn_vae_datasets_v1.py): Combined dataloader wrappers.
  - [unified_graph_dataset_v1.py](file:///home/ntsoi/all_swm/stars/src/dataloaders/unified_graph_dataset_v1.py): Combined SEAN graph dataset.
  - [sean_graph_dataset_v4.py](file:///home/ntsoi/all_swm/stars/src/dataloaders/sean_graph_dataset_v4.py): SEAN-only graph dataset.
  - [socialnav_hetero_graph_dataset_v1.py](file:///home/ntsoi/all_swm/stars/src/dataloaders/socialnav_hetero_graph_dataset_v1.py): SNSB heterogeneous graph dataset.
- **`src/probes/`**: Linear probes for the downstream classification tasks.
- **`src/train_unified_gnn_vae_v3.py`**: Trainer script for the combined model.
- **`src/train_gnn_vae_v4.py`**: Trainer script for the SEAN-only model.
- **`src/eval_label_efficiency.py`**: Downstream label-efficiency sweep script for the combined model.
- **`src/eval_sean_label_efficiency_v4.py`**: Downstream label-efficiency sweep script for the SEAN-only model.
- **`src/summarize_label_efficiency.py`**: Summarizes LP sweep metrics.
- **`src/optuna_unified_v1.py`**: Parallelized Optuna tuning for the combined model.
- **`src/hetero_scripts/`**: Orchestration shell scripts to automate runs.
