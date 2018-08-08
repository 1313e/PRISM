#! /bin/bash
#SBATCH -N 1 -n 16
#SBATCH -J JF12
#SBATCH --time=2-00:00:00
#SBATCH --mem=96GB
#SBATCH --output output.out
#SBATCH --mail-type=END
#SBATCH --mail-user=evandervelden@swin.edu.au
#SBATCH --account=oz071

. deactivate
. activate py2
export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE
python test_imagine.py jf12 RM jf12_3

