FROM libatomsquip/quip-minimal
MAINTAINER "Tamas K. Stenczel <tks32@cam.ac.uk"

# some needed stuff from apt
RUN apt-get -y update \
 && apt-get upgrade -y \
 && apt-get install -y \
      vim wget git \
      quantum-espresso \
    && rm -rf /var/lib/apt/lists/*

# PIP packages
RUN pip install rdkit-pypi flake8 pytest pytest-cov click pyyaml spglib yappi pylint black isort

# DEV use: oh-my-zsh instead of bash
# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t robbyrussell

ENTRYPOINT "/usr/bin/zsh"

