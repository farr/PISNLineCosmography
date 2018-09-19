#!/bin/bash

set -e

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

if [ $# -ne 2 ]; then
  echo "USAGE: run_small.sh NSAMP NSEL"
  exit 1
fi

NSAMP=$1
shift
NSEL=$1
shift

source activate

srun fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_free.h5 --tracefile traceplot_small_free.pdf --prior free && \
srun fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_Planck_Om_w.h5 --tracefile traceplot_small_Planck_Om_w.pdf --prior Planck-Om-w && \
srun fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_H0.h5 --tracefile traceplot_small_H0.pdf --prior H0 && \
srun fit.py --sampfile parameters_small.h5 --samp $NSAMP --selfile selected.h5 --nsel $NSEL --chainfile population_small_H0_Planck_Om.h5 --tracefile traceplot_small_H0_Planck_Om.pdf --prior H0-Planck-Om
