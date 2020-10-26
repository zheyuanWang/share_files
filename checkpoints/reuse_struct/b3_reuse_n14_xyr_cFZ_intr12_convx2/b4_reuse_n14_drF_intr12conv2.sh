# Exit immediately if a comm24and exits with a non-zero status.
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

#backbone_batchsize_datasetVersion_NrOfTest_NrOfParallelLayers_intervarBetweenFrames_ifLoadCheckpoint_NrOfFrames e.p. m3l_b1_range1_n3_int3_ck0_f3
#xyr_combine, centered image after resize& zoom, use checkpoint "reuse_struct/baser1_train_350k/model.ckpt-350000""
EXP_DIR="/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/\
m3l_xyr_cutnet2_centerFZ/b3_reuse_n14_xyr_cFZ_intr12_convx2"

  # if error = Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint
  # : change the model dir where you save your old checkpoints.

cd "${CURRENT_DIR}"

mkdir -p "${EXP_DIR}/train"
cp "${WORK_DIR}"/"${0}" "${EXP_DIR}"/.

#CUDA_VISIBLE_DEVICES=0
python "${WORK_DIR}"/train.py \
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
  --output_stride=8 \
  --decoder_output_stride=8 \
  --train_batch_size=3 \
  --training_number_of_steps=150000 \
  --if_training=1 \
    --log_steps=10 \
    --save_interval_secs=3000 \
    --save_summaries_secs=3000 \
--number_parallel_layers=14 \
--interval_between_frames=11 \
--number_past_frames=2 \
--way_of_combine="OR1x1convx2" \
--mobilenet_struct="reuse_variables" \
--max_scale_factor=1.2 \
--min_scale_factor=0.8 \
          --always_place_image_at_center=True \
  --fine_tune_batch_norm=false \
  --train_logdir="${EXP_DIR}/train" \
  --dataset_dir="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_2f_xyr_intr12/train/" \
  --tf_initial_checkpoint="/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/reuse_struct/b4_reuse_n16_xyr_cFZ_intr12_OR1x1convx2_250k_IOU4197/train/model.ckpt-250000"

    #--tf_initial_checkpoint="/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/reuse_struct/baser1_train_350k/model.ckpt-350000"

  #--tf_initial_checkpoint="/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/reuse_struct/reuse_n11_f2_250k/model.ckpt-250000"


# set FLIP probablity in research/deeplab/input_preprocess.py
# para _PROB_OF_FLIP = 0.5 by default
