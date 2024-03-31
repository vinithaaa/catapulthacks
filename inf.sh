#!/bin/bash

#SBATCH --account=mlp\-n
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=150GB

module load cuda cudnn anaconda

conda activate hack

python codellmpipeline.py
