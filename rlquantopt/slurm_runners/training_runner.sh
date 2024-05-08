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

SCRIPT_PATH=$USER_DIR/rlquantopt_workspace/adace3/token_classification_training.py

python $SCRIPT_PATH $DATASET_DIR $OUTPUT_DIR --lr 1e-5 --max_steps 1500 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --eval_steps 50 --save_steps 100 --seed $SEED

