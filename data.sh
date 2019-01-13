#!/bin/bash
#SBATCH -J tt
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 60:00:00
#SBATCH --gres=gpu:1
module	load	anaconda3/5.3.0	
python /cluster/home/it_stu49/2048/generate_fingerprint.py
