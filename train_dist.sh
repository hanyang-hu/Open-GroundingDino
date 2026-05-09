# #!/bin/bash
# GPU_NUM=$1
# CFG=$2
# DATASETS=$3
# OUTPUT_DIR=$4
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/checkpoints/groundingdino_swint_ogc.pth"}
# TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/checkpoints/bert-base-uncased"}
# echo "
# GPU_NUM = $GPU_NUM
# CFG = $CFG
# DATASETS = $DATASETS
# OUTPUT_DIR = $OUTPUT_DIR
# NNODES = $NNODES
# NODE_RANK = $NODE_RANK
# PORT = $PORT
# MASTER_ADDR = $MASTER_ADDR
# PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
# TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
# "

# # Change ``pretrain_model_path`` to use a different pretrain.
# # (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# # If you don't want to use any pretrained model, just ignore this parameter.

# python -m torch.distributed.launch  --nproc_per_node="${GPU_NUM}" main.py \
#         --output_dir "${OUTPUT_DIR}" \
#         -c "${CFG}" \
#         --datasets "${DATASETS}"  \
#         --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
#         --options text_encoder_type="$TEXT_ENCODER_TYPE"

CFG=$1
DATASETS=$2
OUTPUT_DIR=$3
EXTRA_ARGS="${@:4}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"./checkpoints/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"./checkpoints/bert-base-uncased"}

python main.py --config_file "${CFG}" --datasets "${DATASETS}" --output_dir "${OUTPUT_DIR}" --pretrain_model_path "${PRETRAIN_MODEL_PATH}" --options text_encoder_type="${TEXT_ENCODER_TYPE}" ${EXTRA_ARGS}


# python main.py --config_file config/cfg_odvg.py --datasets config/datasets_mixed_odvg.json --output_dir output --pretrain_model_path content/groundingdino_swint_ogc.pth --options text_encoder_type=content/bert-base-uncased

# With tensorboard:
# python main.py --config_file config/cfg_odvg.py --datasets config/datasets_mixed_odvg.json --output_dir output --pretrain_model_path content/groundingdino_swint_ogc.pth --options text_encoder_type=content/bert-base-uncased --use_tensorboard --tensorboard_dir output/tensorboard --amp
# python -m tensorboard.main --logdir output/tensorboard --port 6006
# python -m tensorboard.main --logdir "C:\Users\kohom\Downloads\Open-GroundingDino\output\tensorboard" --port 6006

# Inference: 
# python tools/inference_on_a_image.py -c config/cfg_odvg.py -p output/checkpoint_best_regular.pth -i data/surgical_instrument/valid/Replicator_02_rgb_4997.png --closed-set -t "overholt clamp . dissecting scissors . ligature clamp . needle holder . peritoneum clamp . surgical scissors ." -o output/pred_images

