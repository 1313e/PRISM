#! /bin/bash
#SBATCH -N 1 -n 4
#SBATCH -J CMF
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --output output.out
#SBATCH --mail-type=END
#SBATCH --mail-user=evandervelden@swin.edu.au
#SBATCH --account=oz071

. deactivate
. activate py2
export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE
mpirun -n 4 python -m mpi4py test_imagine.py cmf full cmf

