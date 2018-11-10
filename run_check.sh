#!/bin/bash
#SBATCH -n 1 --ntasks 6 --cpus-per-task 4 -p cca -o logs/run_check.out -e logs/run_check.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=128
NSEL=2048
./fit.py --sampfile observations.h5 --subset small --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_check.h5 --tracefile traceplot_small_check.pdf > logs/small_check.out 2> logs/small_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset small --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_cosmo_check.h5 --tracefile traceplot_small_cosmo_check.pdf > logs/small_cosmo_check.out 2> logs/small_cosmo_check.err &

NSAMP=512
NSEL=131072
./fit.py --sampfile observations.h5 --subset 1yr --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_check.h5 --tracefile traceplot_1yr_check.pdf > logs/1yr_check.out 2>logs/1yr_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --subset 1yr --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_cosmo_check.h5 --tracefile traceplot_1yr_cosmo_check.pdf > logs/1yr_cosmo_check.out 2>logs/1yr_cosmo_check.err &

NSAMP=128
NSEL=524288
./fit.py --sampfile observations.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_check.h5 --tracefile traceplot_5yr_check.pdf > logs/5yr_check.out 2> logs/5yr_check.err &
./fit.py --cosmo-constraints --sampfile observations.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_cosmo_check.h5 --tracefile traceplot_5yr_cosmo_check.pdf > logs/5yr_cosmo_check.out 2> logs/5yr_cosmo_check.err &

wait
