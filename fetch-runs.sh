#!/bin/bash

set -e 

rsync -e ssh -avz --progress --include="**/*.h5" --include="**/*.pdf" --include="**/*.out" --include="**/logs" --include="**/GWTC-1" --exclude="**" rusty:~/PISN*/ ./
