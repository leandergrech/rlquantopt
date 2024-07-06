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

sleep 1000

