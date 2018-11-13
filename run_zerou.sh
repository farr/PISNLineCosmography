#!/bin/bash
#SBATCH -n 1 --ntasks 6 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=4096
./fit.py --sampfile observations.h5 --subset small --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_small_zerou.h5 --tracefile traceplot_small_zerou.pdf > logs/small_zerou.out 2> logs/small_zerou.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset small --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_small_cosmo_zerou.h5 --tracefile traceplot_small_cosmo_zerou.pdf > logs/small_cosmo_zerou.out 2> logs/small_cosmo_zerou.err &

NSEL=65536
./fit.py --sampfile observations.h5 --subset 1yr --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_zerou.h5 --tracefile traceplot_1yr_zerou.pdf > logs/1yr_zerou.out 2>logs/1yr_zerou.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset 1yr --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo_zerou.h5 --tracefile traceplot_1yr_cosmo_zerou.pdf > logs/1yr_cosmo_zerou.out 2>logs/1yr_cosmo_zerou.err &

NSEL=524288
./fit.py --sampfile observations.h5 --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_zerou.h5 --tracefile traceplot_5yr_zerou.pdf > logs/5yr_zerou.out 2> logs/5yr_zerou.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --zero-uncert --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo_zerou.h5 --tracefile traceplot_5yr_cosmo_zerou.pdf > logs/5yr_cosmo_zerou.out 2> logs/5yr_cosmo_zerou.err &

wait
