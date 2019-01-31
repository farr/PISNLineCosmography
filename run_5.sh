#!/bin/bash
#SBATCH -n 1 --ntasks 4 --cpus-per-task 4 -p cca -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=32768
./fit.py --sampfile observations.h5 --livetime 0.5 --nsamp nsamp_5yr.txt --event-begin 927 --event-end 1835 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_1.h5 --tracefile traceplot_1yr_1.pdf > logs/1yr_1.out 2>&1 &
./fit.py --sampfile observations.h5 --livetime 0.5 --nsamp nsamp_5yr.txt --event-begin 1835 --event-end 2743 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_2.h5 --tracefile traceplot_1yr_2.pdf > logs/1yr_2.out 2>&1 &
./fit.py --sampfile observations.h5 --livetime 0.5 --nsamp nsamp_5yr.txt --event-begin 2743 --event-end 3651 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_3.h5 --tracefile traceplot_1yr_3.pdf > logs/1yr_3.out 2>&1 &
./fit.py --sampfile observations.h5 --livetime 0.5 --nsamp nsamp_5yr.txt --event-begin 3651 --event-end 4558 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_4.h5 --tracefile traceplot_1yr_4.pdf > logs/1yr_4.out 2>&1 &

wait
