# Hetero GNN-VAE Reproducibility Runbook

This runbook is for the current hetero pipeline only.

## Pipeline Files

**Models**
- `models/gnn_vae_v4.py` — SEAN-only v4 model
- `models/unified_gnn_vae_v1.py` — combined SEAN+SNSB model. HeteroMPLayer refines edge embeddings per layer (see Eq. 8 in the paper) to support edge-level downstream probes.

**Dataloaders**
- `dataloaders/unified_graph_dataset_v1.py` — combined SEAN graph dataset
- `dataloaders/unified_gnn_vae_datasets_v1.py` — combined SEAN+SNSB dataloader wrappers
- `dataloaders/sean_graph_dataset_v4.py` — SEAN-only v4 graph dataset
- `dataloaders/socialnav_hetero_graph_dataset_v1.py` — SNSB hetero graph dataset

**Training**
- `train_unified_gnn_vae_v3.py` — combined SEAN+SNSB VAE trainer
- `train_gnn_vae_v4.py` — SEAN-only v4 VAE trainer
- `train_linear_probe_v4.py` — SEAN-only v4 linear probe
- `train_snsb_linear_probe_v3.py` — combined SNSB linear probe

**Eval / Sweep**
- `eval_label_efficiency.py` — combined SEAN+SNSB label-efficiency sweep
- `eval_sean_label_efficiency_v4.py` — SEAN-only v4 label-efficiency sweep
- `summarize_label_efficiency.py` — label-efficiency results summarizer

**Optuna**
- `optuna_unified_v1.py` — combined SEAN+SNSB Optuna search
- `optuna_search_v4.py` — SEAN-only v4 Optuna search

**Shell Scripts**
- `run_train_vae.sh` — train N independent VAEs on a fixed data split
- `run_label_eff.sh` — label-efficiency LP sweep
- `run_summarize.sh` — merge CSVs and summarize
- `run_optuna.sh` — parallel Optuna search

All 19 files above have `Heterogeneous GNN-VAE pipeline` in their header comment. Everything else in `src/` belongs to other pipelines (homogeneous, RF, SWM, SSUB).

**Exploratory (not part of the published pipeline)** — Cocktail Party transfer probe lives under `probes/`, `run_cocktail_probe.py`, and `dataloaders/{hdf5_cocktail_party,cocktail_party_to_hdf5}.py`. STARS pretrained on SEAN+SNSB does not transfer well to this far-OOD static edge-level task; the code is retained for future work but is not invoked by any pipeline script above.

---

- Combined model: SEAN + SNSB unified hetero GNN-VAE.
- SEAN-only model: v4 hetero GNN-VAE with robot, follower, and bystander nodes.
- GPU selection is always via `CUDA_VISIBLE_DEVICES`; the shell wrappers mask each child process to one physical GPU.
- The wrappers schedule at most one process per GPU.
- SEAN surprise is flipped by default where labels are used.
- Default hyperparameters are the selected 256 hidden / 128 latent Optuna configuration.

## Data Files

The three HDF5 files are not in git. To regenerate:

```bash
# SEAN-only
python sean_to_hdf5_v2.py --sean_root SEAN_TOGETHER_Dataset/SEAN_TOGETHER --output dataloaders/sean_together_v2.h5

# SNSB
python socialnav_to_hdf5_v3.py convert --output dataloaders/social_nav_v3.h5

# Combined (SEAN + SNSB, used by the unified model)
python social_nav_converter_unified.py --mode both
```

The fixed scene-level split (`checkpoints/split_indices_final.pt`) is independent of HDF5 regeneration — it does not need to be recreated.

## Server

- SSH: `toberoi@robovisionc`
- Venv: `/scratch/toberoi/swm_venv`

Run all commands from the source directory on your GPU server:

```bash
cd ~/social_world_model/social_world_model/src
source /path/to/your/venv/bin/activate
```

## Script Order

The usual order is:

1. `run_train_vae.sh`
2. `run_label_eff.sh`
3. `run_summarize.sh`
4. `run_tsne.sh` if generating figures

`run_optuna.sh` is optional and only needed if redoing hyperparameter search.

All scripts support `-h` / `--help`.

## Combined Model

Train 10 independent unified SEAN+SNSB VAEs on the fixed data split (random weight initializations, no seeds):

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_train_vae.sh \
  --pipeline combined \
  --out-root checkpoints/repro_v3_hetero_optuned128
```

Run the 10 VAE x 10 LP/subset-draw x 6 fraction label-efficiency sweep:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_v3_hetero_optuned128 \
  --out-prefix label_eff_results_repro_combined
```

Summarize the combined results:

```bash
./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_combined \
  --merged label_eff_results_repro_combined.csv
```

Expected merged size: `1201` lines = header + `10 VAEs * 6 fractions * 10 runs * 2 tasks`.

## SEAN-Only Model

Train 10 independent SEAN-only v4 VAEs on the fixed data split (random weight initializations, no seeds):

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_train_vae.sh \
  --pipeline sean \
  --out-root checkpoints/repro_v4_sean_optuned128
```

Run the 10 VAE x 10 LP/subset-draw x 6 fraction SEAN label-efficiency sweep:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline sean \
  --vae-root checkpoints/repro_v4_sean_optuned128 \
  --out-prefix label_eff_results_repro_sean
```

Summarize the SEAN-only results:

```bash
./hetero_scripts/run_summarize.sh \
  --pipeline sean \
  --in-prefix label_eff_results_repro_sean \
  --merged label_eff_results_repro_sean.csv
```

Expected merged size: `601` lines = header + `10 VAEs * 6 fractions * 10 runs`.

## Probe-Only Ablations

All label-efficiency ablations keep the VAE checkpoint fixed and only change the probe input or prediction baseline.

### Combined Ablations

Random encoder:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_v3_hetero_optuned128 \
  --out-prefix label_eff_results_repro_random_encoder \
  --extra "--mode random_encoder"

./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_random_encoder \
  --merged label_eff_results_repro_random_encoder.csv
```

Weighted random baseline:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_v3_hetero_optuned128 \
  --out-prefix label_eff_results_repro_wrb \
  --extra "--mode weighted_random_baseline"

./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_wrb \
  --merged label_eff_results_repro_wrb.csv
```

Robot-only latent:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_v3_hetero_optuned128 \
  --out-prefix label_eff_results_repro_robot_only \
  --extra "--ablation robot_only"

./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_robot_only \
  --merged label_eff_results_repro_robot_only.csv
```

Bystander-only latent:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline combined \
  --vae-root checkpoints/repro_v3_hetero_optuned128 \
  --out-prefix label_eff_results_repro_bystander_only \
  --extra "--ablation bystander_only"

./hetero_scripts/run_summarize.sh \
  --pipeline combined \
  --in-prefix label_eff_results_repro_bystander_only \
  --merged label_eff_results_repro_bystander_only.csv
```

### SEAN-Only Raw Robot Ablation

Raw robot 592D probes normalized robot input features directly:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_label_eff.sh \
  --pipeline sean \
  --vae-root checkpoints/repro_v4_sean_optuned128 \
  --out-prefix label_eff_results_repro_sean_raw_robot \
  --extra "--ablation raw_robot"

./hetero_scripts/run_summarize.sh \
  --pipeline sean \
  --in-prefix label_eff_results_repro_sean_raw_robot \
  --merged label_eff_results_repro_sean_raw_robot.csv
```

`no_follower` and `zero_follower` are not part of this reproducibility wrapper set.

## t-SNE Figures

Use one representative combined VAE run unless intentionally sweeping all runs.

Ground-truth-colored test t-SNE:

```bash
CUDA_VISIBLE_DEVICES=5 ./hetero_scripts/run_tsne.sh \
  --vae-dir checkpoints/repro_v3_hetero_optuned128/run0 \
  --split test \
  --space encoder \
  --color-by truth
```

Probe-prediction-colored test t-SNE with the same encoder coordinate space:

```bash
CUDA_VISIBLE_DEVICES=5 ./hetero_scripts/run_tsne.sh \
  --vae-dir checkpoints/repro_v3_hetero_optuned128/run0 \
  --split test \
  --space encoder \
  --color-by probe_preds
```

Train split t-SNE:

```bash
CUDA_VISIBLE_DEVICES=5 ./hetero_scripts/run_tsne.sh \
  --vae-dir checkpoints/repro_v3_hetero_optuned128/run0 \
  --split train \
  --space encoder \
  --color-by truth
```

Important flags:

- `--space encoder`: t-SNE coordinates are from VAE encoder embeddings.
- `--space probe_logits`: t-SNE coordinates are from linear-probe logits.
- `--space probe_probs`: t-SNE coordinates are from sigmoid probe probabilities.
- `--space probe_preds`: t-SNE coordinates are from thresholded 0/1 probe predictions.
- `--color-by truth`: point colors are ground-truth binarized labels.
- `--color-by probe_preds`: point colors are thresholded linear-probe predictions.

For the paper-style comparison where positions stay fixed and only labels change, use `--space encoder` for both plots and only change `--color-by`.

## Optuna Best Parameters

### Heterogeneous model (256×128, Trial 10 of `unified_v3_256x128_run1`)

| Param | Value |
|---|---|
| `hidden_dim` | 256 |
| `latent_dim` | 128 |
| `batch_size` | 32 |
| `num_mp_layers` | 2 |
| temporal edge transformer depth | 3 |
| temporal edge transformer heads | 4 |
| `vae_lr` | 1.1564e-4 |
| `beta_max` | 0.9 |
| `warmup_epochs` | 25 |
| `dropout` | 0.2 |
| `adaptive_warmup_epochs` | 20 |
| `adaptive_max_alpha` | 0.75 |
| `adaptive_ramp_epochs` | 100 |

These are the defaults baked into `run_train_vae.sh`. Probe lr is not optuned — fixed at default `1e-3`.

### Homogeneous model (256×128)

| Param | Value |
|---|---|
| `hidden_dim` | 256 |
| `latent_dim` | 128 |
| `batch_size` | 32 |
| `dropout` | 0.1 |
| `weight_decay` | 1e-4 |
| `beta_max` | 0.1 |
| `lr` | 1e-3 |
| transformer layers | 3 |
| `num_mp_layers` | 2 |
| temporal edge transformer depth | 3 |
| temporal edge transformer heads | 4 |
| `adaptive_warmup_epochs` | 20 |
| `adaptive_max_alpha` | 0.75 |
| `adaptive_ramp_epochs` | 100 |

---

## Optional Optuna

The current Optuna scripts select trials using VAE validation loss only. Downstream probe validation loss/AUC is not part of the Optuna objective. The final paper setup tuned the combined SEAN+SNSB model; the SEAN-only baseline reused the selected combined 256x128 hyperparameters where applicable.

Combined Optuna search space: `vae_lr`, `beta_max` in `[0.1, 1.0]`, `warmup_epochs`, `dropout`, `adaptive_warmup_epochs`, `adaptive_max_alpha`, and `adaptive_ramp_epochs` with ramp choices `[50, 100]`.

Run a combined Optuna search:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./hetero_scripts/run_optuna.sh \
  --pipeline combined \
  --study repro_unified_256x128 \
  --storage sqlite:///optuna_repro_unified_256x128.db \
  --n-trials 10
```

`--n-trials` is per worker, so total trials are `n_trials * number_of_visible_GPUs`.

Default fixed dimensions:

- `--hidden-dim 256`
- `--latent-dim 128`

## Common Flags

`run_train_vae.sh`:

- `--pipeline combined|sean`
- `--n-vaes`
- `--out-root`
- `--epochs`
- `--hidden-dim`
- `--latent-dim`
- `--lr`
- `--beta-max`
- `--warmup-epochs`
- `--dropout`
- `--batch-size`
- `--extra "..."`

`run_label_eff.sh`:

- `--pipeline combined|sean`
- `--n-vaes`
- `--vae-root`
- `--vae-prefix`
- `--out-prefix`
- `--no-flip-surprise` for SEAN-only only
- `--extra "..."`

Useful `--extra` values:

- `--mode random_encoder`
- `--mode weighted_random_baseline`
- `--ablation robot_only`
- `--ablation bystander_only`
- `--ablation raw_robot` for SEAN-only only

`run_summarize.sh`:

- `--pipeline combined|sean`
- `--in-prefix`
- `--vae-prefix`
- `--n-vaes`
- `--merged`
- `--task sean|snsb|both`

`run_tsne.sh`:

- `--vae-dir`
- `--vae-root`
- `--n-vaes`
- `--split all|train|val|test`
- `--space encoder|probe_logits|probe_probs|probe_preds`
- `--color-by truth|probe_preds`
- `--perplexity`
- `--out`

## Output Metrics

`summarize_label_efficiency.py` reports mean and standard deviation across runs in each label-fraction cell.

Combined:

- SEAN: `f1_macro`, `auc_macro`, per-label F1/AUC/accuracy.
- SNSB: `f1_macro3` (mean of f1_follow/f1_avoid/f1_not_consider), `f1_macro` (5-class sklearn macro), `acc`, `pa`, per-class F1/precision/accuracy.

SEAN-only:

- `f1_macro`, `auc_macro`, `acc_macro`, per-label F1/AUC/accuracy.
