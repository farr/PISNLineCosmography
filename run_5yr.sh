#!/bin/bash
#SBATCH -n 1 --ntasks 4 --cpus-per-task 4 -t 30-00:00:00 -o logs/5yr.out -e logs/5yr.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=64
NSEL=131072

./fit.py --sampfile parameters.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_free.h5 --tracefile traceplot_5yr_free.pdf --prior free &
./fit.py --sampfile parameters.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_Planck_Om_w.h5 --tracefile traceplot_5yr_Planck_Om_w.pdf --prior Planck-Om-w &
./fit.py --sampfile parameters.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_H0.h5 --tracefile traceplot_5yr_H0.pdf --prior H0 &
./fit.py --sampfile parameters.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_5yr_H0_Planck_Om.h5 --tracefile traceplot_5yr_H0_Planck_Om.pdf --prior H0-Planck-Om &

wait
