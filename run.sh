#!/bin/bash
#SBATCH -n 1 --ntasks 5 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=4096
./fit.py --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small_$NSEL.nc --tracefile traceplot_small_$NSEL.pdf > logs/small_$NSEL.out 2>&1 &

NSEL=32768
./fit.py --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_$NSEL.nc --tracefile traceplot_1yr_$NSEL.pdf > logs/1yr_$NSEL.out 2>&1 &
./fit.py --cosmo-prior --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo_$NSEL.nc --tracefile traceplot_1yr_cosmo_$NSEL.pdf > logs/1yr_cosmo_$NSEL.out 2>&1 &

NSEL=131072
./fit.py --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_$NSEL.nc --tracefile traceplot_5yr_$NSEL.pdf > logs/5yr_$NSEL.out 2>&1 &
./fit.py --cosmo-prior --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo_$NSEL.nc --tracefile traceplot_5yr_cosmo_$NSEL.pdf > logs/5yr_cosmo_$NSEL.out 2>&1 &

wait
