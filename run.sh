#!/bin/bash
#SBATCH -n 1 --ntasks 6 --cpus-per-task 4 -p gen -t 21-00:00:00 -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=4096
./fit.py --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small.h5 --tracefile traceplot_small.pdf > logs/small.out 2>&1 &
./fit.py --cosmo-prior --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small_cosmo.h5 --tracefile traceplot_small_cosmo.pdf > logs/small_cosmo.out 2>&1 &

NSEL=32768
./fit.py --iter 8000 --thin 8 --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf > logs/1yr.out 2>&1 &
./fit.py --iter 8000 --thin 8 --cosmo-prior --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo.h5 --tracefile traceplot_1yr_cosmo.pdf > logs/1yr_cosmo.out 2>&1 &

NSEL=131072
./fit.py --iter 2000 --thin 2 --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf > logs/5yr.out 2>&1 &
./fit.py --cosmo-prior --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo.h5 --tracefile traceplot_5yr_cosmo.pdf > logs/5yr_cosmo.out 2>&1 &

wait
