#!/bin/bash

export ASE_CONFIG_PATH=${HOME}/.config/ase/pytest.config.ini

# Aims
pytest -v -s -rxXs  ../calculators/test_aims.py

