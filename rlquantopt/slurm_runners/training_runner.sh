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

# Initialize variable to indicate slurm is set by default
slurm_set=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in

    --no-slurm)
      slurm_set=false
      ;;
    
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done

# Check if --no-slurm was set
if [ "$slurm_set" = true ]; then
    echo "Slurm mode: preparing environment..."

    # Setup conda environment from requirements.txt if it doesn't exist already
    ENV_NAME=rlquantopt
    USER_DIR=/opt/users/lgrec12
    CONDA_DIR=/opt/local/data/lgrec12/.conda/envs
    PROJ_DIR=$USER_DIR/rlquantopt_workspace/rlquantopt
    PYTHON=python

    # Show some details
    scontrol --details show jobs $SLURM_JOBID |grep RES
    env | grep CUDA

    source /opt/conda/etc/profile.d/conda.sh
    VENV=$CONDA_DIR/$ENV_NAME

    if [ -d $VENV ]; then
        conda activate $ENV_NAME
        echo Conda environment $ENV_NAME activated
    else
        echo Virtual environment $ENV_NAME NOT found
        conda create --name $ENV_NAME python=3.10
        conda activate $ENV_NAME
        conda install pip
    fi

    # Install project dependencies
    REQ_PATH=$PROJ_DIR/requirements.txt
    echo Updating requirements from $REQ_PATH
    pip install -r $REQ_PATH

    # Install RLQuantOpt package in editable mode
    pip uninstall rlquantopt
    pip install -e $PROJ_DIR
else
    echo "Slurm is not set, doing nothing."
fi

# Training script
SCRIPT_PATH=$PROJ_DIR/rlquantopt/rl_agents/train_zcqpee_sb3.py

# Grid-search parameters
PULSE_LENGTHS=(500 1000 3000 6000)
A_NORM_MAXS=(2 5 10)
ACTION_SCALES=(1e-1 1)
T=300
FID_THRESH=0.995
SEEDS=(123 234 345 456 567)

# Training parameters
N_TRAIN=500000
SAVE_FREQ=1000
EVAL_FREQ=500
LOG_INTERVAL=100

# Grid-search
for pl in "${PULSE_LENGTHS[@]}"; do
    for a_max in "${A_NORM_MAXS[@]}"; do
        for a_scale in "${ACTION_SCALES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                $PYTHON $SCRIPT_PATH --n-train $N_TRAIN --save-freq $SAVE_FREQ --eval-freq $EVAL_FREQ --log-interval $LOG_INTERVAL \
                --pulse-length $pl --a-norm-max $a_max --a-scale $a_scale -T $T --fid-thresh $FID_THRESH --seed $seed
            done
        done
    done
done

