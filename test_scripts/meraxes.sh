#! /bin/bash
#SBATCH -N 1 -n 4
#SBATCH -J MERAXES
#SBATCH --time=12:00:00
#SBATCH --mem=22GB
#SBATCH --output output_meraxes.out
#SBATCH --mail-type=END
#SBATCH --mail-user=evandervelden@swin.edu.au
#SBATCH --account=oz071

export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE
mpirun -n 4 python -m mpi4py test_meraxes.py

