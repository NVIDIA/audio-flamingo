#!/bin/bash

ALL_TASK=""
ALL_TASK="${ALL_TASK} MMAU/test"
ALL_TASK="${ALL_TASK} MusicCaps-AudioCaptioning/test"
ALL_TASK="${ALL_TASK} audiocaps-AudioCaptioning/test"

for task in ${ALL_TASK}
do 
    OUTFOLDER=${task//\//-}  # replace / into -
    mkdir -p ../outputs/$OUTFOLDER
done

temp=0.0
numbeams=1
ckpt=-1  # last checkpoint

NAME="sft"
EXP="${NAME}.yaml"

for task in ${ALL_TASK}
do
    echo "task: $task, config: $NAME, ckpt: $ckpt"

    python -u inference.py \
        -c ../configs/$EXP \
        -t $task \
        -temp $temp \
        -nb $numbeams \
        --ckpt ${ckpt}

done