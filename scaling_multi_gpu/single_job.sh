#!/bin/bash
#SBATCH --job-name=single_gpu_cifar100
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --output=logs/single_gpu_cifar100-%j.out

module load scicomp-python-env

python single_gpu.py
