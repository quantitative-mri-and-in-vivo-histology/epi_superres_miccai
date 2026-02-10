FROM python:3.11-slim

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# Install Node.js 20.x
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
  apt-get install -y nodejs && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install all development tools in one layer
# Use --no-install-recommends to avoid pulling in python3.13 as dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  wget \
  build-essential \
  gfortran \
  libopenblas-dev \
  liblapack-dev \
  pkg-config \
  bc \
  dc \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure /usr/bin/python3 points to our Python 3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3 && \
  ln -sf /usr/local/bin/python3.11 /usr/bin/python && \
  ln -sf /usr/local/bin/pip3 /usr/bin/pip3 && \
  ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Install Miniforge (conda-forge default, no ToS)
RUN ARCH=$(uname -m) && \
  if [ "$ARCH" = "x86_64" ]; then \
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
  else \
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
  fi && \
  wget "$MINIFORGE_URL" -O /tmp/miniforge.sh && \
  bash /tmp/miniforge.sh -b -p /opt/conda && \
  rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:$PATH"

# Install MRtrix3 and ANTs
RUN conda install -c mrtrix3 -c conda-forge mrtrix3 ants -y && \
  conda clean -afy

# Create node user (UID 1000 to match typical dev environments)
ARG USERNAME=node
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
  useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
  echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME

# Ensure node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

# Persist bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /workspace/epi1 /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

# Install git-delta
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Set up non-root user
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set the default shell to zsh rather than sh
ENV SHELL=/bin/zsh

# Set the default editor and visual
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
RUN npm install -g @openai/codex

# Switch to root to install Python packages system-wide
USER root

# Install uv for Python package management as root
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  mv /root/.local/bin/uv /usr/local/bin/uv

# Copy requirements.txt and install Python packages during build
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt && \
  rm /tmp/requirements.txt

# Install antspyx/antspynet separately with binary-only (avoid building ITK from source)
RUN uv pip install --system --only-binary :all: antspyx antspynet

# Switch back to node user
USER node

# Add PATH and PYTHONPATH to shell configs
RUN echo 'export PATH="/usr/local/share/npm-global/bin:/usr/local/bin:$PATH"' >> ~/.bashrc && \
  echo 'export PATH="/usr/local/share/npm-global/bin:/usr/local/bin:$PATH"' >> ~/.zshrc && \
  echo 'export PYTHONPATH="/workspace:/workspace/submodules/epi1:$PYTHONPATH"' >> ~/.bashrc && \
  echo 'export PYTHONPATH="/workspace:/workspace/submodules/epi1:$PYTHONPATH"' >> ~/.zshrc

# Configure matplotlib for VS Code plot viewer
USER root
RUN mkdir -p /home/node/.config/matplotlib
COPY .devcontainer/matplotlibrc /home/node/.config/matplotlib/matplotlibrc
RUN chown -R node:node /home/node/.config
USER node

# Copy and set up firewall script
USER root
COPY .devcontainer/init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall

# Copy JAX GPU setup script and epi1 install script
COPY .devcontainer/setup-jax-gpu.sh /workspace/.devcontainer/
COPY .devcontainer/install-epi1.sh /workspace/.devcontainer/
RUN chmod +x /workspace/.devcontainer/setup-jax-gpu.sh \
    /workspace/.devcontainer/install-epi1.sh
USER node
