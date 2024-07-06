#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=research_gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=600
# job parameters
#SBATCH --job-name=rlquantopt-training
#SBATCH --account=rlquantopt
# email user with progress
# SBATCH --mail-user=leander.grech@um.edu.mt
# SBATCH --mail-type=all
#
echo Running on $(hostname)
scontrol --details show jobs $SLURM_JOBID |grep RES
env | grep CUDA

source /opt/conda/etc/profile.d/conda.sh
ENV_NAME=rlquantopt
VENV=/opt/local/data/lgrec12/.conda/envs/$ENV_NAME

USER_DIR=/opt/users/lgrec12
PROJ_DIR=$USER_DIR/rlquantopt_workspace/rlquantopt

if [ -d $VENV ]; then
	conda activate $ENV_NAME
	echo Conda environment $ENV_NAME activated
else
	echo Virtual environment $ENV_NAME NOT found
	conda create --name $ENV_NAME python=3.10
	conda activate $ENV_NAME
	conda install pip
fi
REQ_PATH=$PROJ_DIR/requirements.txt
echo Updating requirements from $REQ_PATH
pip install -r $REQ_PATH
pip install -e $PROJ_DIR

SCRIPT_PATH=$PROJ_DIR/rlquantopt/rl_agents/train_zcqpee_sb3.py

PULSE_LENGTHS=(500 1000 3000 60000)
A_NORM_MAXS=(2 5 10)
ACTION_SCALES=(1e-1 1)
SEEDS=(123 234 345 456 567)
T=300

for pl in "${PULSE_LENGTHS[@]}"; do
  for amax in "${A_NORM_MAXS[@]}"; do
    for ascale in "${ACTION_SCALES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python $SCRIPT_PATH --n-train 500000 --log-interval 100 --save-freq 1000 --eval-freq 500 -T $T --pulse-length $pl  --a-norm-max $amax --a-scale $ascale --seed $seed
      done
    done
  done
done

