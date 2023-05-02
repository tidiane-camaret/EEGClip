### How to use the cluster

# start an interactive session
srun -p ml_gpu-rtx2080 --time=1:00:00 --pty bash

# run scripts
python3 -m scripts.clip.clip_tuh_eeg