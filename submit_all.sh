#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_16.out -e logs/1yr_16.err fit.py --sampfile parameters_1yr.h5 --samp 16 --selfile selected.h5 --chainfile population_1yr_0016.h5 --tracefile traceplot_1yr_0016.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_32.out -e logs/1yr_32.err fit.py --sampfile parameters_1yr.h5 --samp 32 --selfile selected.h5 --chainfile population_1yr_0032.h5 --tracefile traceplot_1yr_0032.pdf --iter 4000 --thin 2

# Five year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_16.out -e logs/5yr_16.err fit.py --sampfile parameters.h5 --samp 16 --selfile selected.h5 --chainfile population_5yr_0016.h5 --tracefile traceplot_5yr_0016.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_32.out -e logs/5yr_32.err fit.py --sampfile parameters.h5 --samp 32 --selfile selected.h5 --chainfile population_5yr_0032.h5 --tracefile traceplot_5yr_0032.pdf
