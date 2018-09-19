#!/bin/bash

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

COMMON_ARGS="-n 1 --tasks-per-node=4 --cpus-per-task 4 --exclusive=user"
CCA_ARGS="$COMMON_ARGS -p cca"
GEN_ARGS="$COMMON_ARGS -t 20-00:00:00"

# Small runs
NSEL=8192
NSAMP=64
sbatch $CCA_ARGS -o logs/small.out -e logs/small.err run_small.sh $NSAMP $NSEL

# One-year runs
NSEL=32768
sbatch $CCA_ARGS -o logs/1yr.out -e logs/1yr.err run_1yr.sh $NSAMP $NSEL

# Five year runs.  These are *long* (28 days estimated), so go in the general queue.
NSEL=131072
sbatch $GEN_ARGS -o logs/5yr.out -e logs/5yr.err run_5yr.sh $NSAMP $NSEL
