### script to run few shot classification experiments
# lms : medicalai/ClinicalBERT
: '
# lm lr tuning
for text_encoder_name in bert-base-uncased 
do  
    for lr_frac_lm in 0 0.1 0.001 0.0001 1
    do  

        /home/jovyan/test_env/bin/python /home/jovyan/EEGClip/scripts/clip/train_eegclip_tuh.py  --text_encoder_name $text_encoder_name --lr_frac_lm $lr_frac_lm --n_epochs 5 --batch_size 8

    done
done 

# category tuning
(
'
categories=("all" "IMPRESSION" "DESCRIPTION OF THE RECORD" \
                          "CLINICAL HISTORY" "MEDICATIONS" "INTRODUCTION" \
                          "CLINICAL CORRELATION" "HEART RATE" "FINDINGS" "REASON FOR STUDY" \
                          "TECHNICAL DIFFICULTIES" "EVENTS" "CONDITION OF THE RECORDING"\
                          "PAST MEDICAL HISTORY" "TYPE OF STUDY" "ACTIVATION PROCEDURES"\
                          "NOTE")
for category in "${categories[@]}"
do 
    /home/jovyan/test_env/bin/python /home/jovyan/EEGClip/scripts/clip/train_eegclip_tuh.py --category "$category" --n_epochs 2

done
