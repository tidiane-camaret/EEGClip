### Run on Nail Cluster
python3 -m scripts.clip.clip_tuh_eeg --n_rec 2993 --nailcluster


### Run on KISlurm

# information about resources
sinfo
sfree

# start an interactive session
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash 

# other request with larger resources : 
srun -p ml_gpu-rtx2080 -c 20 --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --gres=gpu:2 --time=3:00:00 --pty bash 


# tmux
tmux new-session -s <name>
# detach: 
ctrl+b d
# attach: 
tmux attach -t <name>

# run scripts
cd ~/dev/neuro_ai/EEGClip
python3 -m scripts.clip.clip_tuh_eeg 


# start a job
sbatch scripts/cluster/job.txt

# see all jobs
sacct --user=$USER

# see all running jobs
squeue --user=$USER

# see job details
scontrol show job 3868830

# cancel job
scancel 3868830