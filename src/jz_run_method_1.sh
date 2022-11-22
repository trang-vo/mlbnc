#!/bin/bash
#SBATCH --job-name=TrainRL
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=TrainRL%j.out
#SBATCH --error=TrainRL%j.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -A vws@v100
#SBATCH --exclusive
#SBATCH --hint=nomultithread


module purge
module load python/3.7.10
conda activate trang

python train_agent.py tsp subtour PriorCutEnv