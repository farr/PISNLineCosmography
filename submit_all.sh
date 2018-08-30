#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_128.out -e logs/1yr_128.err fit.py --sampfile parameters_1yr.h5 --samp 128 --selfile selected.h5 --chainfile population_1yr_0128.h5 --tracefile traceplot_1yr_0128.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_256.out -e logs/1yr_256.err fit.py --sampfile parameters_1yr.h5 --samp 256 --selfile selected.h5 --chainfile population_1yr_0256.h5 --tracefile traceplot_1yr_0256.pdf

# Five year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_128.out -e logs/5yr_128.err fit.py --sampfile parameters.h5 --samp 128 --selfile selected.h5 --chainfile population_5yr_0128.h5 --tracefile traceplot_5yr_0128.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_256.out -e logs/5yr_256.err fit.py --sampfile parameters.h5 --samp 256 --selfile selected.h5 --chainfile population_5yr_0256.h5 --tracefile traceplot_5yr_0256.pdf
