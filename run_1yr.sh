#!/bin/bash
#SBATCH -n 1 --ntasks 1 --cpus-per-task 4 -p cca -o logs/1yr.out -e logs/1yr.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=128
NSEL=32768

./fit.py --sampfile parameters_1yr.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf &

wait
