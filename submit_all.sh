#!/bin/bash

set -e

source ~/.bashrc

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_16.out -e logs/1yr_16.err fit.py --sampfile parameters_1yr.h5 --samp 16 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_1yr_0016.h5 --tracefile traceplot_1yr_0016.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_32.out -e logs/1yr_32.err fit.py --sampfile parameters_1yr.h5 --samp 32 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_1yr_0032.h5 --tracefile traceplot_1yr_0032.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_64.out -e logs/1yr_64.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_1yr_0064.h5 --tracefile traceplot_1yr_0064.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_128.out -e logs/1yr_128.err fit.py --sampfile parameters_1yr.h5 --samp 128 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_1yr_0128.h5 --tracefile traceplot_1yr_0128.pdf


# Five year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_16.out -e logs/5yr_16.err fit.py --sampfile parameters.h5 --samp 16 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_5yr_0016.h5 --tracefile traceplot_5yr_0016.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_32.out -e logs/5yr_32.err fit.py --sampfile parameters.h5 --samp 32 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_5yr_0032.h5 --tracefile traceplot_5yr_0032.pdf
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_64.out -e logs/5yr_64.err fit.py --sampfile parameters.h5 --samp 64 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_5yr_0064.h5 --tracefile traceplot_5yr_0064.pdf
# We are starting to get long runs here; should take about 8 days, set 10 to be safe, which means we have to 
# use the 'gen' queue
sbatch -n 1 --cpus-per-task 4 -p cca -t 14-00:00:00 -o logs/5yr_128.out -e logs/5yr_128.err fit.py --sampfile parameters.h5 --samp 128 --selfile selected.h5 --stanfile PISNLineCosmography.stan --chainfile population_5yr_00128.h5 --tracefile traceplot_5yr_00128.pdf
