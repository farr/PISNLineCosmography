#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
  echo "USAGE: swap_to_checks.py XXX"
  echo "will swap population_XXX.h5 to population_XXX_check.h5, etc"
  exit 1
fi

l=$1
shift

mv population_${l}.h5 population_${l}_check.h5
mv traceplot_${l}.pdf traceplot_${l}_check.pdf
mv logs/${l}.out logs/${l}_check.out
mv logs/${l}.err logs/${l}_check.err
