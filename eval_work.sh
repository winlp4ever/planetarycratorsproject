#!/usr/bin/env bash
pathToObjDetec=/media/redlcamille/NewVol/studies/tensorflow/models/research/object_detection
MODEL_DIR=models/
#PIPELINE_CONFIG_PATH=models/config/faster_rcnn_resnet101_coco.config
PIPELINE_CONFIG_PATH=models/config/ssd_mobilenet_v1_coco.config
python $pathToObjDetec/legacy/eval.py --logtostderr \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --checkpoint_dir $MODEL_DIR/train/ \
    --eval_dir $MODEL_DIR/eval/