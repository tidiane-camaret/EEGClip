### script to run few shot classification experiments

for fold_idx in 0 1 2 3 4 
do
    :' #train the EEGClip model 
    python -m scripts.clip.train_eegclip_tuh --fold_idx $fold_idx
    '
    for task_name in 'pathological' 'gender' 'under_50' 'epilep' 'seizure' 'medication'
    do  

        # evaluate the EEGClip model (frozen encoder)
        python -m scripts.classif.classification_tuh \
                    --task_name $task_name \
                    --freeze_encoder \
                    --fold_idx $fold_idx

        # evaluate a frozen model
        python -m scripts.classif.classification_tuh \
                    --task_name $task_name \
                    --freeze_encoder \
                    --weights random \
                    --fold_idx $fold_idx

        # evaluate a fully trainable model 
        python -m scripts.classif.classification_tuh \
                    --task_name $task_name \
                    --weights random \
                    --fold_idx $fold_idx
        # evaluate a frozen model trained on a different task
        :'
        python -m scripts.classif.classification_tuh \
                    --task_name $task_name \
                    --weights pathological_task \
                    --freeze_encoder \

        python -m scripts.classif.classification_tuh \
                    --task_name $task_name \
                    --weights under_50_task \
                    --freeze_encoder \
        '
    done 
done
