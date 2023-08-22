#!/bin/bash

module load vasp

export GAP_RSS_TEST_SETUP=${PWD}/setup_rss_test

# rm -r $GAP_RSS_TEST_SETUP
if [ -e $GAP_RSS_TEST_SETUP ]; then
    if [ ! -d $GAP_RSS_TEST_SETUP ]; then
        echo "GAP_RSS_TEST_SETUP=$GAP_RSS_TEST_SETUP but is not a directory" 1>&2
        exit 1
    fi
    echo "WARNING: run dir $GAP_RSS_TEST_SETUP exists, trying to resume from current point." 1>&2
    echo "Type enter to continue or ^C to abort if you need to delete it" 1>&2
    read dummy
else
    mkdir -p $GAP_RSS_TEST_SETUP
fi

export VASP_COMMAND=vasp.serial
export VASP_COMMAND_GAMMA=vasp.gamma_serial
export PYTEST_VASP_POTCAR_DIR=${VASP_PATH}/pot/rev_54/PBE
export GRIF_BUILDCELL_CMD=${HOME}/src/work/AIRSS/airss-0.9.1/src/buildcell/src/buildcell

nohup pytest -s tests/test_cli_rss.py 1> prep_test_cli_rss.stdout 2> prep_test_cli_rss.stderr &
