#!/bin/bash

set -e

texcount -v3 -merge -incbib -dir -sub=none -utf8 -sum pisn-line.tex
