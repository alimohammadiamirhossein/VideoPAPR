#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1


cd /$project/papr
source /$project/papr/.venv/bin/activate

python3.9 train.py --opt configs/butterfly.yml