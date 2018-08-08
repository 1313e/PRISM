#! /bin/bash
#SBATCH -N 1 -n 4
#SBATCH -J SINE_WAVE
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --output output_sine.out
#SBATCH --mail-type=END
#SBATCH --mail-user=evandervelden@swin.edu.au
#SBATCH --account=oz071

export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE
mpirun -n 4 python -m mpi4py test_sine_wave.py

