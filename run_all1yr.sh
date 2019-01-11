#!/bin/bash
#SBATCH -n 1 --ntasks 4 --cpus-per-task 4 -t 7-00:00:00 -o logs/run.out -e logs/run.err

set -e

source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=65536
./fit.py --sampfile observations.h5 --event-begin 927 --event-end 1835 --livetime 0.5 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_2.h5 --tracefile traceplot_1yr_2.pdf > logs/1yr_2.out 2>logs/1yr_2.err &
./fit.py --sampfile observations.h5 --event-begin 1835 --event-end 2743 --livetime 0.5 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_3.h5 --tracefile traceplot_1yr_3.pdf > logs/1yr_3.out 2>logs/1yr_3.err &
./fit.py --sampfile observations.h5 --event-begin 2743 --event-end 3651 --livetime 0.5 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_4.h5 --tracefile traceplot_1yr_4.pdf > logs/1yr_4.out 2>logs/1yr_4.err &
./fit.py --sampfile observations.h5 --event-begin 3651 --event-end 4558 --livetime 0.5 --selfile selected.h5 --nsel $NSEL --chainfile population_1yr_5.h5 --tracefile traceplot_1yr_5.pdf > logs/1yr_5.out 2>logs/1yr_5.err &

wait
