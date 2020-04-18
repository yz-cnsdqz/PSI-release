#!/bin/bash
source /home/yzhang/virtualenv/python3-torch-1.2/bin/activate
# module load cuda/10.0
# module load cudnn/7.5-cu10.0

PROJECT_PATH=/home/yzhang/workspaces/smpl-env-gen-3d-internal
GEN_PATH=/home/yzhang/workspaces/smpl-env-gen-3d-internal/results_prox_stage1_nosceneloss/virtualrealcams
FIT_PATH=/home/yzhang/workspaces/smpl-env-gen-3d-internal/results_prox_stage1_nosceneloss_postproc/virtualrealcams

python -u ${PROJECT_PATH}/source/fitting_proxe.py ${GEN_PATH} ${FIT_PATH} \
	2>&1 | tee -a fittinginfo_results_prox_stage1_nosceneloss_postproc.txt


# deactivate
