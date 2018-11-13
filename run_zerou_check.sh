#!/bin/bash
#SBATCH -n 1 --ntasks 6 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=8192
./fit.py --sampfile observations.h5 --subset small --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_small_zerou_check.h5 --tracefile traceplot_small_zerou_check.pdf > logs/small_zerou_check.out 2> logs/small_zerou_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset small --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_small_cosmo_zerou_check.h5 --tracefile traceplot_small_cosmo_zerou_check.pdf > logs/small_cosmo_zerou_check.out 2> logs/small_cosmo_zerou_check.err &

NSEL=131080
./fit.py --sampfile observations.h5 --subset 1yr --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_zerou_check.h5 --tracefile traceplot_1yr_zerou_check.pdf > logs/1yr_zerou_check.out 2>logs/1yr_zerou_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset 1yr --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo_zerou_check.h5 --tracefile traceplot_1yr_cosmo_zerou_check.pdf > logs/1yr_cosmo_zerou_check.out 2>logs/1yr_cosmo_zerou_check.err &

NSEL=1048640
./fit.py --sampfile observations.h5 --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_zerou_check.h5 --tracefile traceplot_5yr_zerou_check.pdf > logs/5yr_zerou_check.out 2> logs/5yr_zerou_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo_zerou_check.h5 --tracefile traceplot_5yr_cosmo_zerou_check.pdf > logs/5yr_cosmo_zerou_check.out 2> logs/5yr_cosmo_zerou_check.err &

wait
