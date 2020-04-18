#!/bin/bash
source /is/ps2/yzhang/virtualenv/python3-torch-1.2/bin/activate
module load cuda/10.0
module load cudnn/7.5-cu10.0

model_type=$1
prox_data=$2
use_scene_loss=$3


PROJECT_PATH=/is/ps2/yzhang/workspaces/PSI-internal
BATCHSIZE=32
EPOCH=30
LRH=0.0003
WeightLossVPoser=0.001
WeightLossKL=0.01
WeightLossContact=0.000001
WeightLossCollision=0.000001


if [ "$use_scene_loss" == 1 ]         
then                                                                                                                                                                                                   
	WeightLossContact=0.01
	WeightLossCollision=0.1
fi

use_all=0
if [ "$prox_data" == all ]         
then                                                                                                                                                                                                   
	use_all=1
fi


CKPT_PATH=checkpoints_prox${prox_data}_model${model_type}_batch${BATCHSIZE}_epoch${EPOCH}_LR${LRH}_LossVposer${WeightLossVPoser}_LossKL${WeightLossKL}_LossContact${WeightLossContact}_LossCollision${WeightLossCollision}
OUTDIR=${PROJECT_PATH}/checkpoints/${CKPT_PATH}
TRAINLOG=${PROJECT_PATH}/trainlogs

python -u ${PROJECT_PATH}/source/train_${model_type}.py --save_dir ${OUTDIR} --batch_size ${BATCHSIZE} \
								--lr_s ${LRH} --lr_h ${LRH} --num_epoch ${EPOCH} \
								--weight_loss_vposer ${WeightLossVPoser} \
								--weight_loss_kl ${WeightLossKL} \
								--weight_loss_contact ${WeightLossContact} \
								--weight_loss_collision ${WeightLossCollision} \
								--use_all ${use_all} \
	2>&1 | tee -a ${TRAINLOG}/traininfo_${CKPT_PATH}.txt

deactivate
