#!/bin/bash
#SBATCH -o /data/users2/bbaker/projects/dynib/slurm//%j.out
#SBATCH -e /data/users2/bbaker/projects/dynib/slurm//%j.err
#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --mem 8G
#SBATCH --gres=gpu:V100:1
#SBATCH -p qTRDGPUH
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --oversubscribe
#SBATCH -t 7200
eval "$(conda shell.bash hook)"
args="${@:1}"
echo args $args
conda activate catalyst
cd /data/users2/bbaker/projects/dynib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
PYTHONPATH=. python experiment.py $args
