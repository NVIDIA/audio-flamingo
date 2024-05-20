#!/bin/bash

YOUR_DATA_ROOT_DIR=YOUR_DATA_ROOT_DIR
EXP=foundation_sft_8_shot.yaml

ALL_TASK=""
ALL_TASK="${ALL_TASK} Clotho-v2-AudioCaptioning/test"
ALL_TASK="${ALL_TASK} CochlScene-SceneClassification/test"
ALL_TASK="${ALL_TASK} NonSpeech7k-EventClassification/test"
ALL_TASK="${ALL_TASK} audiocaps-AudioCaptioning/interleaved_knn-test"
ALL_TASK="${ALL_TASK} Medley-solos-DB-InstrClassification/interleaved_knn-test"

for TASK in ${ALL_TASK}
do 
    OUTFOLDER=${TASK//\//-}  # replace / into -
    mkdir -p ../outputs/$OUTFOLDER
done

TEMP=0.0
NUMBEAMS=3
CKPT=99

L=${#EXP}
NAME=$(echo ${EXP} | cut -c 1-$(($L-5)))  # remove last .yaml

for TASK in ${ALL_TASK}
do
    echo "task: $TASK, config: $NAME, checkpoint: $CKPT"

    OUTFOLDER=${TASK//\//-}
    OUTFILE="../outputs/$OUTFOLDER/$NAME-CKPT${CKPT}.log"
    CKPT_DIR="$YOUR_DATA_ROOT_DIR/audio-flamingo-data/checkpoint/$NAME"

    if [[ -f "$CKPT_DIR/checkpoint_${CKPT}.pt" ]]; then 
        python -u inference.py \
            -c ../configs/$EXP \
            -t $TASK \
            -TEMP $TEMP \
            -nb $NUMBEAMS \
            --CKPT ${CKPT} >> $OUTFILE
    fi 
done
