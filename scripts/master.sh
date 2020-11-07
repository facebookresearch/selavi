# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --gres=gpu:8
#SBATCH --job-name=SeLaVi_pretrain
#SBATCH --mem=450GB
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/%u/jobs/%j.out
#SBATCH --signal=USR1@120
#SBATCH --time=72:00:00


source activate lab_vid

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES


master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

if [ -z "$1" ]
then
	DATASET="kinetics"
else
	DATASET=$1
fi
if [ -z "$2" ]
then
	NUM_CLUSTERS=400
else
	NUM_CLUSTERS=$2
fi
if [ -z "$3" ]
then
	HEADCOUNT=1
else
	HEADCOUNT=$3
fi
if [ -z "$4" ]
then
	MATCH="True"
else
	MATCH=$4
fi
if [ -z "$5" ]
then
	DISTRIBUTION="default"
else
	DISTRIBUTION=$5
fi
if [ -z "$6" ]
then
	IND_GROUPS=1
else
	IND_GROUPS=$6
fi

### DATA PARAMS
ROOT_DIR="/path/to/kinetics"
BATCH_SIZE=16
NUM_FRAMES=30
TARGET_FPS=30
SAMPLE_RATE=1
NUM_CLIPS=1
CROP_SIZE=112
AUD_NUM_SEC=1
AUD_SAMPLE_RATE=24000
AUD_SPEC_TYPE=2
AUD_VOLUME_JITTERING='True'
AUD_TEMPORAL_JITTERING='False'
AUD_Z_NORMALIZE='False'

### OPTIM PARAMS
EPOCHS=201
BASE_LR=1e-2
WEIGHT_DECAY=1e-5
LR_WARM_EPOCHS=10
USE_WAMRUP='True'
USE_LR_SCHEDULER='False'

### MODEL PARAMS
VID_BASE_ARCH='r2plus1d_18'
AUD_BASE_ARCH='resnet9'
USE_MLP='True'

### EXP DUMP PATH
EXP_NAME="dataset_${DATASET}_clusters_${NUM_CLUSTERS}_headcount_${HEADCOUNT}_match_${MATCH}_dis_${DISTRIBUTION}_ind_groups_${IND_GROUPS}"
SAV_FOLDER="./experiments/SeLaVi/${EXP_NAME}"
mkdir -p ${SAV_FOLDER}

# command
srun --label python -u main.py \
--root_dir ${ROOT_DIR} \
--dump_path ${SAV_FOLDER} \
--num_frames ${NUM_FRAMES} \
--target_fps ${TARGET_FPS} \
--sample_rate ${SAMPLE_RATE} \
--num_train_clips ${NUM_CLIPS} \
--train_crop_size ${CROP_SIZE} \
--num_sec_aud ${AUD_NUM_SEC} \
--aud_sample_rate ${AUD_SAMPLE_RATE} \
--aud_spec_type ${AUD_SPEC_TYPE} \
--use_volume_jittering ${AUD_VOLUME_JITTERING} \
--use_audio_temp_jittering ${AUD_TEMPORAL_JITTERING} \
--z_normalize ${AUD_Z_NORMALIZE} \
--epochs ${EPOCHS} \
--batch_size ${BATCH_SIZE} \
--base_lr ${BASE_LR} \
--wd ${WEIGHT_DECAY} \
--warmup_epochs ${LR_WARM_EPOCHS} \
--use_warmup_scheduler ${USE_WAMRUP} \
--use_lr_scheduler ${USE_LR_SCHEDULER} \
--vid_base_arch ${VID_BASE_ARCH} \
--aud_base_arch ${AUD_BASE_ARCH} \
--use_mlp ${USE_MLP} \
--mlp_dim ${NUM_CLUSTERS} \
--dist_url $dist_url \
--ds_name ${DATASET} \
--headcount ${HEADCOUNT} \
--match ${MATCH} \
--distribution ${DISTRIBUTION} \
--ind_groups ${IND_GROUPS} \
