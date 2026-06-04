FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

RUN pip install --no-cache-dir \
    torch-geometric \
    pyyaml \
    h5py \
    numpy \
    matplotlib \
    scikit-learn \
    wandb \
    optuna

WORKDIR /workspace
