#!/bin/bash
#SBATCH -p ml_gpu-rtx2080 # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G # memory pool for all cores
#SBATCH -o scripts/cluster/logs/%x.%N.%j.out # STDOUT file  (the folder log has to be created prior to running or this won't work, %x %N %j will be replaced by jobname hostname jobid)
#SBATCH -e scripts/cluster/logs/%x.%N.%j.err # STDERR file (the folder log has to be created prior to running or this won't work)
#SBATCH -J clip_tuh_eeg_job # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
 
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
 
# Job to perform, here for example you would call your pytho nfile that runs your code
## run training script
 python3 -m scripts.clip.clip_tuh_eeg --n_recordings_to_load 2993 #--string_sampling
## run sweep
# wandb agent ayousyku
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";