#!/bin/bash
#SBATCH --job-name=model_test
#SBATCH --output=%j_model_test.out
#SBATCH --error=%j_model_test.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=20G
export PATH=/share/home/lsy_chenyanchao/software/miniconda3/bin:$PATH

source activate deeplearning

EXEC=/share/home/lsy_chenyanchao/projects/model_test/scr/main.py

echo "process will start at : "
date

python3 $EXEC --config ../config.yaml --mode optimize

echo "process end at : "
date
