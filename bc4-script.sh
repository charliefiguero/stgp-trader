#!/usr/bin/env bash
#SBATCH --partition cpu
#SBATCH --time 0-6:00
#SBATCH --mem 60GB


# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
pip install deap --user
pip install jsonpickle --user
pip install pygraphviz --user

python BSE2.py

echo "done"
