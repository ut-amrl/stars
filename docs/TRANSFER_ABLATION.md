# Cross-Dataset Transfer Ablation (single-dataset pretraining)

This ablation tests whether the STARS-He latent captures **shared, dataset-agnostic**
social structure by pretraining on **one** dataset and evaluating downstream on the
**other**:

- **SNSB → SEAN**: pretrain on SocialNav-SUB only, probe SEAN.
- **SEAN → SNSB**: pretrain on SEAN only, probe SocialNav-SUB.

Because the heterogeneous model has *dataset-specific* input adapters (per-dataset stems
and temporal edge encoders) feeding a *shared* trunk (message passing + `fc_mu`), a model
pretrained on one dataset leaves the other dataset's adapter at random initialization.
Evaluating the held-out dataset through that random adapter is confounded. The ablation
therefore uses a two-stage protocol:

1. **Stage A — single-dataset pretraining.** Train the unified VAE with the held-out
   dataset's loss weight set to 0 (so only one dataset shapes the shared latent).
2. **Stage B — adapter alignment.** Freeze the shared trunk *and* the trained dataset's
   adapter; train **only** the held-out dataset's input adapter (stems + edge encoder +
   its decoder head) with the **same self-supervised reconstruction loss** (recon + β·KL,
   no labels). This maps the held-out dataset into the frozen single-dataset latent
   without altering the latent geometry.

The downstream label-efficiency probe and summarization then reuse the standard STARS
pipeline unchanged.

## Files

- `src/train_stage_b_adapter_align.py` — Stage-B adapter-alignment trainer.
- `src/hetero_scripts/run_stageA_transfer.sh` — launches both single-dataset Stage-A
  pretrains (SEAN-only + SNSB-only), fanned across a GPU pool.
- `src/hetero_scripts/run_stageB.sh` — fans Stage-B alignment across the per-VAE run dirs.

Everything else is reused from the main pipeline: `train_unified_gnn_vae_v3.py`
(Stage A, via the existing `--snsb_loss_weight` / `--sean_loss_weight` flags),
`run_label_eff.sh`, `run_summarize.sh`.

## Workflow

All commands run from `src/`. Hyperparameters match the STARS-He 256×128 config.

### 1. Stage A — single-dataset pretraining (10 VAEs each)

```bash
# SEAN-only (GPUs 2-4) and SNSB-only (GPUs 5-7), 10 VAEs each, in parallel:
./hetero_scripts/run_stageA_transfer.sh
# -> checkpoints/xfer_seanonly/run{0..9}   (SEAN-only)
# -> checkpoints/xfer_snsbonly/run{0..9}   (SNSB-only)
```

Equivalently, a single direction via the main wrapper:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hetero_scripts/run_train_vae.sh --pipeline combined \
  --n-vaes 10 --out-root checkpoints/xfer_seanonly --extra "--snsb_loss_weight 0"
```

### 2. Stage B — align the held-out adapter

Align the adapter for the dataset the model was **not** pretrained on:

```bash
# SEAN-only pretrain -> align the SNSB adapter (for SNSB downstream)
./hetero_scripts/run_stageB.sh --align snsb \
  --vae-root checkpoints/xfer_seanonly --out-root checkpoints/xfer_seanonly_alignSNSB

# SNSB-only pretrain -> align the SEAN adapter (for SEAN downstream)
./hetero_scripts/run_stageB.sh --align sean \
  --vae-root checkpoints/xfer_snsbonly --out-root checkpoints/xfer_snsbonly_alignSEAN
```

Each aligned run dir contains `best_unified_gnn_vae_v3_seednone.pt` + a copied
`split_used_v3_seednone.pt`, i.e. a drop-in input for `run_label_eff.sh`.

### 3. Label-efficiency eval + summarize

```bash
./hetero_scripts/run_label_eff.sh --pipeline combined \
  --vae-root checkpoints/xfer_seanonly_alignSNSB --out-prefix label_eff_seanonly_to_snsb
./hetero_scripts/run_summarize.sh --pipeline combined \
  --in-prefix label_eff_seanonly_to_snsb --merged label_eff_seanonly_to_snsb.csv
# rename the generic summaries so the other direction doesn't overwrite them:
mv summary_snsb.csv summary_snsb__seanonly_to_snsb.csv
mv summary_sean.csv summary_sean__seanonly_to_snsb.csv
```

Repeat with `--vae-root checkpoints/xfer_snsbonly_alignSEAN --out-prefix label_eff_snsbonly_to_sean`.

## Reading the result

`run_label_eff.sh` evaluates **both** SEAN and SNSB tasks from each checkpoint (rows are
tagged by a `task` column; `run_summarize.sh` splits them into `summary_sean.csv` /
`summary_snsb.csv`). The **transfer** number is the *opposite*-dataset column:

- `xfer_snsbonly_alignSEAN` → the **SEAN** summary is the transfer result (SNSB→SEAN);
  the SNSB summary is the in-domain reference.
- `xfer_seanonly_alignSNSB` → the **SNSB** summary is the transfer result (SEAN→SNSB);
  the SEAN summary is the in-domain reference.

For SNSB, report `f1_macro3` (mean F1 of avoid/follow/not_consider) to match the paper's
Table 1 — the 5-class `f1_macro` is lower-bounded by overtake/yield having ~0 support.
