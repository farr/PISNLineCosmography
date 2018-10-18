#!/bin/bash
#SBATCH -n 1 --ntasks 1 --cpus-per-task 4 -p cca -o logs/5yr.out -e logs/5yr.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=128
NSEL=131072

./fit.py --sampfile parameters.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf &

wait
