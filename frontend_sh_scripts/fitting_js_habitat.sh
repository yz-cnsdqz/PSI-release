#!/bin/bash
source /is/ps2/yzhang/virtualenv/python3-torch-1.2/bin/activate
module load cuda/10.0
module load cudnn/7.5-cu10.0

PROJECT_PATH=/is/ps2/yzhang/workspaces/smpl-env-gen-3d
GEN_PATH=/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_habitat_stage1_nosceneloss/virtualcams
FIT_PATH=/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_habitat_stage1_nosceneloss_postproc/virtualcams

python -u ${PROJECT_PATH}/source/fitting_habitat.py ${GEN_PATH} ${FIT_PATH} \
	2>&1 | tee -a fittinginfo_results_habitat_stage2_sceneloss_postproc.txt


deactivate
