#!/bin/bash
#SBATCH --job-name=StandardTSP
#SBATCH --qos=qos_cpu-dev
#SBATCH --output=StandardTSP%j.out
#SBATCH --error=StandardTSP%j.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH -A vws@cpu

module purge
module load pytorch-gpu/py3/1.7.1

srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 0 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 1 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 2 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 3 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 4 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 5 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 6 &
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run_methods.py 7 &
wait