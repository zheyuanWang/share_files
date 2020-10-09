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

#backbone_batchsize_datasetVersion_NrOfTest_NrOfParallelLayers_intervarBetweenFrames_ifLoadCheckpoint_NrOfFrames e.p. m3l_b1_range1_n3_int3_ck0_f3
EXP_DIR="/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/\
m3l_1x1_xyr_centerFZ/b4_reuse_n16_xyr_cFZ_intr12_OR1x1convx2_fullmask"



cd "${CURRENT_DIR}"

mkdir -p "${EXP_DIR}/val"
cp "${WORK_DIR}"/"${0}" "${EXP_DIR}"/.

CUDA_VISIBLE_DEVICES=-1 python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v3_large_seg" \
  --dataset="skitti_gm" \
  --eval_crop_size="502,1002" \
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
  --checkpoint_dir="${EXP_DIR}/train" \
  --eval_logdir="${EXP_DIR}/val" \
--if_training=0 \
--number_parallel_layers=16 \
--interval_between_frames=11 \
--number_past_frames=2 \
--way_of_combine="OR1x1convx2" \
--mobilenet_struct="reuse_variables" \
--max_scale_factor=1.2 \
--min_scale_factor=0.8 \
          --always_place_image_at_center=True \
  --dataset_dir="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_2f_xyr_intr12/val/" \
  --max_number_of_evaluations=0


#/mrtstorage/users/zwang/ArchivedExperiments/m3l_b4_range20_0/train