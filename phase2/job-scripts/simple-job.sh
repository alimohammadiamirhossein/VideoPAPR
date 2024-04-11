#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=8:30:0
#SBATCH --mail-user=<srr8@sfu.ca>
#SBATCH --mail-type=ALL
cd ~/$projects/papr
module load StdEnv/2020
module load python/3.9
module load gcc/9.3.0
module load opencv/4.5.5
module load rust/1.53.0
source ~/py_papr/bin/activate
date

python train1.py
