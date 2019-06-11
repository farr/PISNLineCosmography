#!/bin/bash
#SBATCH -n 1 --ntasks 3 --cpus-per-task 4 -p cca -o exact.out -e exact.err

set -e
source ~/.bashrc
source activate
export PYTHONPATH="$PYTHONPATH:/mnt/home/wfarr/PISNLineCosmography"

NSEL=32768
python exact_test.py --nsel $NSEL > exact-$NSEL.out 2>&1 &

NSEL=65536
python exact_test.py --nsel $NSEL > exact-$NSEL.out 2>&1 &

NSEL=131072
python exact_test.py --nsel $NSEL > exact-$NSEL.out 2>&1 &

wait
