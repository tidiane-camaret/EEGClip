### script to run few shot classification experiments


for task_name in 'pathological' 'gender' 'under_50' 'epilep' 'seizure' 'medication'
do  
    for train_frac in 1 2 5 10 20 50
    do  
        for seed in 1 2 3 4 5
        do

            # evaluate the EEGClip model (frozen encoder)
            python -m scripts.classif.classification_tuh \
                        --task_name $task_name \
                        --freeze_encoder \
                        --train_frac $train_frac \
                        --seed $seed



            # evaluate a frozen model
            python -m scripts.classif.classification_tuh \
                        --task_name $task_name \
                        --freeze_encoder \
                        --weights random \
                        --train_frac $train_frac \
                        --seed $seed

            # evaluate a fully trainable model 
            python -m scripts.classif.classification_tuh \
                        --task_name $task_name \
                        --weights random \
                        --train_frac $train_frac \
                        --seed $seed

            # evaluate a frozen model trained on a different task

            if [ "$task_name" = "pathological" ]; then
                weights="under_50"
            else
                weights="pathological"
            fi

            python -m scripts.classif.classification_tuh \
                        --task_name $task_name \
                        --weights $weights \
                        --freeze_encoder \
                        --train_frac $train_frac \
                        --seed $seed
            
        done
    done
done 

