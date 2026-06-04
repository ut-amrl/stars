# Build arguments
ARG gid
ARG uid
ARG username

# =============================================================================
# Stage 1: Base - NVIDIA CUDA + Ubuntu with Essential Tools, VNC, and Chrome
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS utilities

ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    ca-certificates \
    sudo \
    locales \
    lsb-release \
    tmux \
    tmuxinator \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Setup Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

ARG TARGETARCH

# Install Python packages.
# amd64: CUDA-enabled PyTorch + matching PyG wheels.
# arm64: CPU fallback wheels.
RUN set -e; \
    if [ "$TARGETARCH" = "amd64" ]; then \
        pip3 install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu128 \
            torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 && \
        pip3 install --no-cache-dir \
            torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv h5py \
            numpy scipy pandas scikit-learn matplotlib seaborn tqdm \
            -f https://data.pyg.org/whl/torch-2.8.0+cu128.html; \
    else \
        pip3 install --no-cache-dir \
            torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
            torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv h5py \
            numpy scipy pandas scikit-learn matplotlib seaborn tqdm \
            -f https://data.pyg.org/whl/torch-2.8.0+cpu.html; \
    fi

# Install XFCE4 and VNC
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_amd64.deb -O /tmp/turbovnc.deb; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_arm64.deb -O /tmp/turbovnc.deb; \
    fi && \
    dpkg -i /tmp/turbovnc.deb && \
    rm /tmp/turbovnc.deb && \
    ln -s /opt/TurboVNC/bin/* /usr/local/bin/

# Copy VNC startup script
COPY scripts/vnc /usr/local/bin/vnc
RUN chmod +x /usr/local/bin/vnc

# Install Chrome (or Chromium on ARM64)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
        apt-get update && apt-get install -y google-chrome-stable; \
    else \
        apt-get update && apt-get install -y chromium-browser; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 10: Final - User Creation and Configuration
# =============================================================================
FROM utilities AS final

ARG uid
ARG gid
ARG username
ARG ros_distro

ENV UID=$uid
ENV GID=$gid
ENV USERNAME=$username
ENV ROS_DISTRO=$ros_distro

# Create user and group, then configure sudo
RUN set -e && \
    # Create group if it doesn't exist
    getent group ${GID} || groupadd -g ${GID} ${USERNAME} && \
    # Handle user creation/modification
    if id ${UID} >/dev/null 2>&1; then \
        # User with this UID exists, modify it
        existing_user=$(id -nu ${UID}) && \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $existing_user 2>/dev/null || true && \
        usermod -g ${GID} ${USERNAME} 2>/dev/null || true; \
    else \
        # Create new user
        useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}; \
    fi && \
    # Ensure home directory exists and has correct ownership
    mkdir -p /home/${USERNAME} && \
    chmod 777 /home/${USERNAME} && \
    (chown -R ${UID}:${GID} /home/${USERNAME} || true) && \
    # Add to sudo group and configure passwordless sudo
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    visudo -c && \
    # Add the user to the video group for access to video devices
    usermod -aG video ${USERNAME}

# Copy bash_profile files from docker directory
COPY docker/bash_profile.shared /tmp/
# Copy setup script
COPY scripts/setup_bash_profile.sh /tmp/
RUN chmod +x /tmp/setup_bash_profile.sh
# Combine shared and platform-specific bash_profile based on architecture
RUN /tmp/setup_bash_profile.sh ${USERNAME} && \
    rm -f /tmp/bash_profile.shared /tmp/setup_bash_profile.sh


# Create .tmux.conf
RUN echo "set -g mouse on" >> /home/${USERNAME}/.tmux.conf && \
    echo "set -g history-limit 10000" >> /home/${USERNAME}/.tmux.conf && \
    (chown ${USERNAME}:${GID} /home/${USERNAME}/.tmux.conf || true)

# Final guard to ensure runtime deps are present in the final image.
RUN pip3 install --no-cache-dir h5py wandb pytorch-ignite

USER $username
WORKDIR /home/$username
