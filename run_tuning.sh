#!/usr/bin/env bash
#PIPELINE_CONFIG_PATH=models/config/faster_rcnn_resnet101_coco.config
PIPELINE_CONFIG_PATH=models/config/ssd_mobilenet_v1_coco.config
MODEL_DIR=models/
NUM_TRAIN_STEPS=$1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
pathToObjDetec=/media/redlcamille/NewVol/studies/tensorflow/models/research/object_detection
python $pathToObjDetec/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=$MODEL_DIR/train \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

#python $pathToObjDetec/legacy/train.py --logtostderr \
#    --train_dir models/train/ \
#    --pipeline_config_path $PIPELINE_CONFIG_PATH