#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
  echo "USAGE: swap_to_checks.py [small|1yr|5yr]"
  exit 1
fi

l=$1
shift

mv population_${l}.h5 population_${l}_check.h5
mv population_${l}_cosmo.h5 population_${l}_cosmo_check.h5
mv traceplot_${l}.pdf traceplot_${l}_check.pdf
mv traceplot_${l}_cosmo.pdf traceplot_${l}_cosmo_check.pdf
mv logs/${l}.out logs/${l}_check.out
mv logs/${l}.err logs/${l}_check.err
mv logs/${l}_cosmo.out logs/${l}_cosmo_check.out
mv logs/${l}_cosmo.err logs/${l}_cosmo_check.err
