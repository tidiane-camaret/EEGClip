### script to run few shot classification experiments


for task_name in 'pathological' 'gender' 'under_50' 'epilep' 'seizure' 'medication'
do  

    # evaluate the EEGClip model (frozen encoder)
    python -m scripts.classif.classification_tuh \
                --task_name $task_name \
                --freeze_encoder \


    # evaluate a frozen model
    python -m scripts.classif.classification_tuh \
                --task_name $task_name \
                --freeze_encoder \
                --weights random \


    # evaluate a fully trainable model 
    python -m scripts.classif.classification_tuh \
                --task_name $task_name \
                --weights random \

    # evaluate a frozen model trained on a different task

    if [$task_name == 'pathological']
    then
    weights = 'under_50_task'
    else
    weights = 'pathological_task'
    fi

    python -m scripts.classif.classification_tuh \
                --task_name $task_name \
                --weights $weights \
                --freeze_encoder \

done 

