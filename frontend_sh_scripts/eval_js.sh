#!/bin/bash
source /is/ps2/yzhang/virtualenv/python3-torch-1.2/bin/activate
module load cuda/10.0
module load cudnn/7.5-cu10.0

PROJECT_PATH=/is/ps2/yzhang/workspaces/smpl-env-gen-3d
GEN_PATH=/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_ckp4_envloss2/realcams
RES_PATH=eval_collision_ckp4_envloss2_realcams.txt

python -u ${PROJECT_PATH}/utils_eval_collision.py ${GEN_PATH} ${RES_PATH}
	

deactivate
