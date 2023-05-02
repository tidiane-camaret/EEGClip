### Run on Nail Cluster
python3 -m scripts.clip.clip_tuh_eeg --n_recordings_to_load 3000 --nailcluster


### Run on KISlurm

# information about resources
sinfo
sfree

# start an interactive session
srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=3:00:00 --pty bash 

# run scripts
cd ~/dev/neuro_ai/EEGClip
python3 -m scripts.clip.clip_tuh_eeg --n_recordings_to_load 2993


# start a job
