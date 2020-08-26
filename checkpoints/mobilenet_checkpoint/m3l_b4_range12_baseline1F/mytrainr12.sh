#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e

#wzy: use the right visual enviroment
. /home/zwang/venv/bin/activate

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/model_test.py

#EXP_DIR folder name's meaning: backbone_batchsize_datasetVersion_NrOfTest
#EXP_DIR="/mrtstorage/users/zwang/experiments/m3l_b4_range5_1/"
EXP_DIR="/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/important/m3l_b4_range12"

cd "${CURRENT_DIR}"

mkdir -p "${EXP_DIR}/train"
cp "${WORK_DIR}"/"${0}" "${EXP_DIR}"/.
CUDA_VISIBLE_DEVICES=1 python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v3_large_seg" \
  --dataset="skitti_gm" \
  --train_crop_size="502,1002" \
  --image_pooling_stride=4,5 \
  --aspp_convs_filters=128 \
  --aspp_with_concat_projection=0 \
  --aspp_with_squeeze_and_excitation=1 \
  --decoder_use_sum_merge=1 \
  --decoder_filters=19 \
  --decoder_output_is_logits=0 \
  --image_se_uses_qsigmoid=1 \
  --image_pyramid=1 \
  --decoder_output_stride=8 \
  --train_batch_size=4 \
--save_interval_secs=7500 \
--save_summaries_secs=7500 \
  --training_number_of_steps=250000 \
  --fine_tune_batch_norm=false \
  --train_logdir="${EXP_DIR}/train" \
  --dataset_dir="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange12/train" \
  --tf_initial_checkpoint="/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_r1_origin_350k_test/backup/model.ckpt-350000"