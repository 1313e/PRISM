#! /bin/bash
#SBATCH -N 1 -n 4
#SBATCH -J CMF_10
#SBATCH --time=24:00:00
#SBATCH --mem=22GB
#SBATCH --output output_$SLURM_ARRAY_TASK_ID.out
#SBATCH --mail-type=END
#SBATCH --mail-user=evandervelden@swin.edu.au
#SBATCH --account=oz071
#SBATCH --array=101-150

. deactivate
. activate py2
export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE
python test_imagine.py cmf full cmf_10_$SLURM_ARRAY_TASK_ID

