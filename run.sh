#!/bin/bash
#SBATCH -n 1 --ntasks 3 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=4096
./fit.py --sampfile observations.h5 --nsamp nsamp_small.txt --subset small --selfile selected.h5 --nsel $NSEL --chainfile population_small.h5 --tracefile traceplot_small.pdf > logs/small.out 2>&1 &

NSEL=32768
./fit.py --sampfile observations.h5 --nsamp nsamp_1yr.txt --subset 1yr --selfile selected.h5 --nsel $NSEL --chainfile population_1yr.h5 --tracefile traceplot_1yr.pdf > logs/1yr.out 2>&1 &

NSEL=131072
./fit.py --sampfile observations.h5 --nsamp nsamp_5yr.txt --selfile selected.h5 --nsel $NSEL --chainfile population_5yr.h5 --tracefile traceplot_5yr.pdf > logs/5yr.out 2>&1 &

wait
