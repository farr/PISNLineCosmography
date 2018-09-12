#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

# Small runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small.out -e logs/small.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 2048 --chainfile population_small.h5 --tracefile traceplot_small.pdf

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr.out -e logs/1yr.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 32768 --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf

# Five year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr.out -e logs/5yr.err fit.py --sampfile parameters.h5 --samp 64 --selfile selected.h5 --nsel 131072 --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf
