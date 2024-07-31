#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=research_gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=3000
# job parameters
#SBATCH --job-name=rlquantopt-training
#SBATCH --account=rlquantopt
# email user with progress
# SBATCH --mail-user=leander.grech@um.edu.mt
# SBATCH --mail-type=all
#
echo Running on $(hostname)

USER_DIR=/home/leander/code
PROJ_DIR=$USER_DIR/rlquantopt
PYTHON=python

# Initialize variable to indicate slurm is set by default
slurm_set=true
start_idx=0
cnt=200


# Loop through all arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --start-idx) # Check if next parameter is set and not another flag
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                start_idx=$2
                shift
            else
                echo "Error: --start-idx requires a numerical argument"
                exit 1
            fi
            ;;
        --cnt) # Check if next parameter is set and not another flag
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                cnt=$2
                shift
            else
                echo "Error: --cnt requires a numerical argument"
                exit 1
            fi
            ;;
        --no-slurm) # Set slurm to false if flag is present
            slurm_set=false
            ;;
        *) # Handle unknown parameters
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

echo ""

# Check if --no-slurm was set
if [ "$slurm_set" = true ]; then
    echo "Slurm mode: preparing environment..."

    # Setup conda environment from requirements.txt if it doesn't exist already
    USER_DIR=/opt/users/lgrec12
    PROJ_DIR=$USER_DIR/rlquantopt_workspace/rlquantopt

    ENV_NAME=rlquantopt
    CONDA_DIR=/opt/local/data/lgrec12/.conda/envs
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
SCRIPT_DIR=$PROJ_DIR/rlquantopt/rl_agents
SCRIPT_PATH=$SCRIPT_DIR/train_zcqpee_sb3.py
#tensorboard --logdir .  &

# Grid-search parameters
N_ENVS=16
PULSE_LENGTHS=(1000 3000 6000)
A_NORM_MAX=10
ACTION_SCALES=(1e-2 1e-1 1)
T=300
FID_THRESH=0.995
SEEDS=(123 234 345 456 567)

# Training parameters
N_TRAIN=1500000
SAVE_FREQ=1000
EVAL_FREQ=500
LOG_INTERVAL=500

# Grid-search
idx=0
limit=$((start_idx + cnt))
for SEED in "${SEEDS[@]}"; do
    for PL in "${PULSE_LENGTHS[@]}"; do
        N_STEPS=$((PL * 2))
        for A_SCALE in "${ACTION_SCALES[@]}"; do
            idx=$((idx + 1))
            if [ "$idx" -lt "$start_idx" ]; then
                continue
            fi
            echo "seed=$SEED, pl=$PL, a_scale=$A_SCALE"
            $PYTHON $SCRIPT_PATH --pulse-length $PL --delta-mode -T $T --a-scale $A_SCALE --a-norm-max $A_NORM_MAX\
            --fid-thresh $FID_THRESH --n-steps $N_STEPS --n-train $N_TRAIN --save-freq $SAVE_FREQ \
            --eval-freq $EVAL_FREQ --log-interval $LOG_INTERVAL --seed $SEED --n-envs $N_ENVS

            if [ "$idx" -gt "$limit" ]; then
                echo "Reached session nb. of runs limit"
                exit 0
            fi
        done
    done
done


