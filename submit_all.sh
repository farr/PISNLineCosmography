#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

# Small runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small.out -e logs/small.err fit.py --sampfile parameters_small.h5 --samp 256 --selfile selected.h5 --nsel 8192 --chainfile population_small.h5 --tracefile traceplot_small.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small2.out -e logs/small2.err fit.py --sampfile parameters_small.h5 --samp 256 --selfile selected.h5 --nsel 8192 --chainfile population_small2.h5 --tracefile traceplot_small2.pdf

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr.out -e logs/1yr.err fit.py --sampfile parameters_1yr.h5 --samp 128 --selfile selected.h5 --nsel 65536 --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr2.out -e logs/1yr2.err fit.py --sampfile parameters_1yr.h5 --samp 128 --selfile selected.h5 --nsel 65536 --chainfile population_1yr2.h5 --tracefile traceplot_1yr2.pdf

# Five year runs.  These are *long* (28 days estimated), so go in the general queue.
sbatch -n 1 --cpus-per-task 4  -t 50-00:00:00 -o logs/5yr.out -e logs/5yr.err fit.py --sampfile parameters.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf
sbatch -n 1 --cpus-per-task 4  -t 50-00:00:00 -o logs/5yr2.out -e logs/5yr2.err fit.py --sampfile parameters.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr2.h5 --tracefile traceplot_5yr2.pdf
