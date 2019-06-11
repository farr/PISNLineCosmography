#!/bin/bash
#SBATCH -n 1 --ntasks 1 --cpus-per-task 4 -p cca -o exact.out -e exact.err

set -e
source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=262144
python exact_test.py --nsel $NSEL > exact-$NSEL.out 2>&1 &

wait
