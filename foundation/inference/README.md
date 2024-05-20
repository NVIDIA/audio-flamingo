# Audio Flamingo Inference

To run inference over a dataset, first add that dataset information into the ```foundation_sft_*.yaml --> data_config --> valid_dataset_config```. If it's an in-domain testset, similar to the validation datasets there, it's enough to write ```<dataset_name>-<flamingo_task>/test: true``` for regular testsets or ```<dataset_name>-<flamingo_task>/interleaved_knn-test: true``` for ICL testsets. If it's a 0-shot testset, you should also add the augmentations to that entry similar to ```dataset_blending_config```, for example:

```
Medley-solos-DB-InstrClassification/interleaved_knn-test:
    prefix_prob: 1.0
    augmentations:
        do_nothing: 1.0
```

Then, the inference code is

```
EXP=<config_filename>.yaml
TASK=<dataset_name>-<flamingo_task>/test or <dataset_name>-<flamingo_task>/interleaved_knn-test
TEMP=<inference_temperature>
NUMBEAMS=<inference_number_of_beams>
CKPT=<integer_checkpoint_number>
OUTFILE=<output_filename>

python -u inference.py \
    -c ../configs/$EXP \
    -t $TASK \
    -temp $TEMP \
    -nb $NUMBEAMS \
    --ckpt $CKPT >> $OUTFILE
```

A sample script to launch all inference code is in ```inference.sh```.