#!/bin/bash
#SBATCH -n 1 --ntasks 1 --cpus-per-task 4 -p cca -o logs/small.out -e logs/small.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=128
NSEL=8192

./fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small.h5 --tracefile traceplot_small.pdf &

wait
