#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

# Small runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small_free.out -e logs/small_free.err fit.py --sampfile parameters_small.h5 --samp 64 --selfile selected.h5 --nsel 8192 --chainfile population_small_free.h5 --tracefile traceplot_small_free.pdf --prior free
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small_Planck_Om_w.out -e logs/small_Planck_Om_w.err fit.py --sampfile parameters_small.h5 --samp 64 --selfile selected.h5 --nsel 8192 --chainfile population_small_Planck_Om_w.h5 --tracefile traceplot_small_Planck_Om_w.pdf --prior Planck-Om-w
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small_H0.out -e logs/small_H0.err fit.py --sampfile parameters_small.h5 --samp 64 --selfile selected.h5 --nsel 8192 --chainfile population_small_H0.h5 --tracefile traceplot_small_H0.pdf --prior H0
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/small_H0_Planck_Om.out -e logs/small_H0_Planck_Om.err fit.py --sampfile parameters_small.h5 --samp 64 --selfile selected.h5 --nsel 8192 --chainfile population_small_H0_Planck_Om.h5 --tracefile traceplot_small_H0_Planck_Om.pdf --prior H0-Planck-Om

# One-year runs
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_free.out -e logs/1yr_free.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 65536 --chainfile population_1yr_free.h5 --tracefile traceplot_1yr_free.pdf --prior free
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_Planck_Om_w.out -e logs/1yr_Planck_Om_w.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 65536 --chainfile population_1yr_Planck_Om_w.h5 --tracefile traceplot_1yr_Planck_Om_w.pdf --prior Planck-Om-w
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_H0.out -e logs/1yr_H0.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 65536 --chainfile population_1yr_H0.h5 --tracefile traceplot_1yr_H0.pdf --prior H0
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/1yr_H0_Planck_Om.out -e logs/1yr_H0_Planck_Om.err fit.py --sampfile parameters_1yr.h5 --samp 64 --selfile selected.h5 --nsel 65536 --chainfile population_1yr_H0_Planck_Om.h5 --tracefile traceplot_1yr_H0_Planck_Om.pdf --prior H0-Planck-Om

# Five year runs.  These are *long* (28 days estimated), so go in the general queue.
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_free.out -e logs/5yr_free.err fit.py --sampfile parameters_5yr.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr_free.h5 --tracefile traceplot_5yr_free.pdf --prior free
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_Planck_Om_w.out -e logs/5yr_Planck_Om_w.err fit.py --sampfile parameters_5yr.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr_Planck_Om_w.h5 --tracefile traceplot_5yr_Planck_Om_w.pdf --prior Planck-Om-w
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_H0.out -e logs/5yr_H0.err fit.py --sampfile parameters_5yr.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr_H0.h5 --tracefile traceplot_5yr_H0.pdf --prior H0
sbatch -n 1 --cpus-per-task 4 -p cca -o logs/5yr_H0_Planck_Om.out -e logs/5yr_H0_Planck_Om.err fit.py --sampfile parameters_5yr.h5 --samp 64 --selfile selected.h5 --nsel 262144 --chainfile population_5yr_H0_Planck_Om.h5 --tracefile traceplot_5yr_H0_Planck_Om.pdf --prior H0-Planck-Om
