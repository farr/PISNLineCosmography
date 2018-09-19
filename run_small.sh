#!/bin/bash
#SBATCH --ntasks 4 --cpus-per-task 4 -p cca -o logs/small.out -e logs/small.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSAMP=64
NSEL=8192

./fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_free.h5 --tracefile traceplot_small_free.pdf --prior free &
./fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_Planck_Om_w.h5 --tracefile traceplot_small_Planck_Om_w.pdf --prior Planck-Om-w &
./fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_H0.h5 --tracefile traceplot_small_H0.pdf --prior H0 &
./fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_H0_Planck_Om.h5 --tracefile traceplot_small_H0_Planck_Om.pdf --prior H0-Planck-Om &

wait
