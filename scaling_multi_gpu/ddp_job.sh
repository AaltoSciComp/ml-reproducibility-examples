#!/bin/bash
#SBATCH --job-name=ddp_cifar100
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --output=logs/ddp_cifar100-%j.out

module load scicomp-python-env

TF_ENABLE_ONEDNN_OPTS=0 OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 ddp.py --seed 1234
