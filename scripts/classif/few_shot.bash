### script to run few shot classification experiments

for task_name in 'pathological' 'gender' 'under_50' 'epilep' 'seizure' 'medication'
do  
    for train_frac in 1 2 5 10 20 50
    do  
        for seed in 1 2 3 4 5
        do
            python -m scripts.classif.few_shot_decoding \
                     --task_name $task_name \
                     --train_frac $train_frac \
                     --freeze_encoder \
                     --seed $seed
            python -m scripts.classif.few_shot_decoding \
                     --task_name $task_name \
                     --train_frac $train_frac \
                     --weights random \
                     --seed $seed
            python -m scripts.classif.few_shot_decoding \
                     --task_name $task_name \
                     --train_frac $train_frac \
                     --freeze_encoder \
                     --weights random \
                     --seed $seed
        done 
    done
done
