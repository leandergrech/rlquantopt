#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=research_gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=250
# job parameters
#SBATCH --job-name=rlquantopt-training
#SBATCH --account=rlquantopt
# email user with progress
#SBATCH --mail-user=leander.grech@um.edu.mt
#SBATCH --mail-type=all
#
echo Running on $(hostname)
scontrol --details show jobs $SLURM_JOBID |grep RES
env | grep CUDA

source /opt/conda/etc/profile.d/conda.sh
ENV_NAME=rlquantopt
VENV=/opt/local/data/lgrec12/.conda/envs/$ENV_NAME
if [ -d $VENV ]; then
	conda activate $ENV_NAME
	echo Conda environment $ENV_NAME activated
else
	echo Virtual environment $ENV_NAME NOT found
#	REQ_PATH=/opt/users/lgrec12/rlquantopt_workspace/requirements.txt
#	echo Creating from $REQ_PATH
	conda create --name $ENV_NAME python=3.10
	conda activate $ENV_NAME
	conda install pip
#	pip install -r $REQ_PATH
fi

USER_DIR=/opt/users/lgrec12
PROJ_DIR=$USER_DIR/rlquantopt_workspace/rlquantopt
pip install $PROJ_DIR
SCRIPT_PATH=$PROJ_DIR/rlquantopt/rl_agents/qpee_sb3_training.py

python $SCRIPT_PATH --n-envs 10 --n-train 1000 --log-interval 1 --save-freq 1000

