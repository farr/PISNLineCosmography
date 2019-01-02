#!/bin/bash
#SBATCH -n 1 --ntasks 3 --cpus-per-task 4 -t 30-00:00:00 -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=2048
./fit.py --sampfile observations.h5 --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small.h5 --tracefile traceplot_small.pdf > logs/small.out 2> logs/small.err &

NSEL=32768
./fit.py --sampfile observations.h5 --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf > logs/1yr.out 2>logs/1yr.err &

NSEL=131072
./fit.py --sampfile observations.h5 --selfile selected.h5 --nsel $NSEL --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf > logs/5yr.out 2> logs/5yr.err &

wait
