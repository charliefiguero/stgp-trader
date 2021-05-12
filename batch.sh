#!/bin/bash
#SBATCH --partition cpu
#SBATCH --time 0-6:00
#SBATCH --mem 80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --output stgp_diss.out


# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load languages/anaconda3/2018.12

pip install deap --user
pip install jsonpickle --user
pip install pygraphviz --user
pip install typing --user

python BSE2.py

echo "done"

