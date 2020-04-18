from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import sys, os, glob
import pdb
import json
import argparse
import numpy as np
import open3d as o3d

sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal')
sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal/source')




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import smplx
from human_body_prior.tools.model_loader import load_vposer
import chamfer_pytorch.dist_chamfer as ext

from cvae import BodyParamParser, GeometryTransformer

from sklearn.cluster import KMeans





def fitting(fittingconfig):

    input_data_file = fittingconfig['input_data_file']
    with open(input_data_file, 'rb') as f:
        body_param_input = pickle.load(f)

    xh, _, _= BodyParamParser.body_params_parse_fitting(body_param_input)

    return xh.detach().cpu().numpy()






if __name__=='__main__':


    gen_path = sys.argv[1]

    if 'proxe' in gen_path:
        scene_test_list = ['MPH16', 'MPH1Library','N0SittingBooth', 'N3OpenArea']
    elif 'habitat' in gen_path:
        scene_test_list = ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge', 
                    '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1', 
                    'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0', 
                    'zsNo4HB9uLZ-livingroom0_13']
    xh_list = []
    for scenename in scene_test_list:
        for ii in range(5000):
            input_data_file = os.path.join(gen_path,scenename+'/body_gen_{:06d}.pkl'.format(ii))
            if not os.path.exists(input_data_file):
                continue

            fittingconfig={
                'input_data_file': input_data_file,
                'scene_verts_path': '/home/yzhang/Videos/PROXE/scenes_downsampled/'+scenename+'.ply',
                'scene_sdf_path': '/home/yzhang/Videos/PROXE/scenes_sdf/'+scenename,
                'human_model_path': '/home/yzhang/body_models/VPoser',
                'vposer_ckpt_path': '/home/yzhang/body_models/VPoser/vposer_v1_0',
                'init_lr_h': 0.1,
                'num_iter': 50,
                'batch_size': 1, 
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'contact_id_folder': '/is/cluster/work/yzhang/PROX/body_segments',
                'contact_part': ['back','butt','L_Hand','R_Hand','L_Leg','R_Leg','thighs'],
                'verbose': False
            }

            xh = fitting(fittingconfig)
            xh_list.append(xh)


    ar = np.concatenate(xh_list, axis=0)

    ## k-means
    import scipy.cluster
    codes, dist = scipy.cluster.vq.kmeans(ar, 20)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
    
    from scipy.stats import entropy
    ee = entropy(counts)
    print('entropy:' + str(ee))
    print('mean distance:' + str(np.mean(dist)) )
