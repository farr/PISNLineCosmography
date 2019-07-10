#!/bin/bash
#SBATCH -n 1 --ntasks 2 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=2048
../fit.py --stanfile ../model.stan --sampfile GWTC-1-observations.h5 --selfile GWTC-1-selected.h5 --nsel $NSEL --nmix 4 --chainfile population_GWTC-1_$NSEL.nc --tracefile traceplot_GWTC-1_$NSEL.pdf > logs/GWTC-1_$NSEL.out 2>&1 &
NSEL=4096
../fit.py --stanfile ../model.stan --sampfile GWTC-1-observations.h5 --selfile GWTC-1-selected.h5 --nsel $NSEL --nmix 4 --chainfile population_GWTC-1_$NSEL.nc --tracefile traceplot_GWTC-1_$NSEL.pdf > logs/GWTC-1_$NSEL.out 2>&1 &

wait
