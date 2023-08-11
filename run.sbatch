#!/bin/bash
#SBATCH --job-name=maze-combo
#SBATCH --partition=gpu-a40
#SBATCH --account=weirdlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/simondu/zhaoyi/OfflineRL-Kit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaoyi20@cs.washington.edu

source ~/.bashrc
conda activate test
# singularity shell --nv --bind /gscratch dta.sif
# source /opt/conda/etc/profile.d/conda.sh
# conda activate dta
# singularity exec --nv --bind /gscratch dta.sif ./run.sh > result/Breakout2.txt

# ./diffusion_2simple.sh
# ./dt_out.sh
./run.sh
# ./run.sh
# ./learn.sh
# ./rollout.sh > rollout_pred_1e6.out
# ./rollout.sh
# python scripts/create_maze_dataset.py --debug
# ./combo.sh
# ./diffusion.sh
# ./diffusion_2simple.sh