#!/bin/bash 
#SBATCH --job-name=lstm
#SBATCH --output=lstm_adam_KDec_5_200output.log
#SBATCH --error=lstm_adam_KDec_5_200error.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1# Request 1 L4 per node
#SBATCH --mem=80G # Request 64GB memory
#SBATCH --time=24:00:00 # Request 12 hours runtime

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

#activate your environment
source /home/bg91882/environments/quixer/bin/activate

module load Python/3.11.3-GCCcore-12.3.0

python run.py -d cuda -m LSTM




