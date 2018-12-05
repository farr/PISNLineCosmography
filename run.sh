#!/bin/bash
#SBATCH -n 1 --ntasks 6 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=8192
./fit.py --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small.h5 --tracefile traceplot_small.pdf --fitfile fit_small.pkl.bz2 > logs/small.out 2> logs/small.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small_cosmo.h5 --tracefile traceplot_small_cosmo.pdf --fitfile fit_small_cosmo.pkl.bz2 > logs/small_cosmo.out 2> logs/small_cosmo.err &

NSEL=131072
./fit.py --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf --fitfile fit_1yr.pkl.bz2 > logs/1yr.out 2>logs/1yr.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo.h5 --tracefile traceplot_1yr_cosmo.pdf --fitfile fit_1yr_cosmo.pkl.bz2 > logs/1yr_cosmo.out 2>logs/1yr_cosmo.err &

NSEL=1048576
./fit.py --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf --fitfile fit_5yr.pkl.bz2 > logs/5yr.out 2> logs/5yr.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo.h5 --tracefile traceplot_5yr_cosmo.pdf --fitfile fit_5yr_cosmo.pkl.bz2 > logs/5yr_cosmo.out 2> logs/5yr_cosmo.err &

wait
