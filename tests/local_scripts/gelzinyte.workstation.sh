#!/bin/bash

export ASE_CONFIG_PATH=${HOME}/.config/ase/pytest.config.ini

# MOPAC isn't updated with the profile
export ASE_MOPAC_COMMAND="${HOME}/programs/mopac-22.1.1-linux/bin/mopac PREFIX.mop 2> /dev/null"

# Aims
pytest -v -s -rxXs  ../calculators/test_aims.py

