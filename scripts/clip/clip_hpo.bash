### script to run few shot classification experiments


for text_encoder_name in bert-base-uncased medicalai/ClinicalBERT
do  
    for lr_frac_lm in 0 0.1 0.001 0.0001 1
    do  

        /home/jovyan/test_env/bin/python /home/jovyan/EEGClip/scripts/clip/train_eegclip_tuh.py  --text_encoder_name $text_encoder_name --lr_frac_lm $lr_frac_lm

    done
done 