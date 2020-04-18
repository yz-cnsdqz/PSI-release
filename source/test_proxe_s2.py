from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import sys, os, glob
import json
import argparse
import numpy as np
import open3d as o3d
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler

import smplx
from human_body_prior.tools.model_loader import load_vposer
import chamfer_pytorch.dist_chamfer as ext

from cvae import BodyParamParser, HumanCVAES2, GeometryTransformer
from batch_gen import BatchGeneratorTest




class TestOP:
    def __init__(self, testconfig):
        for key, val in testconfig.items():
            setattr(self, key, val)

        if not os.path.exists(self.ckpt_dir):
            print('--[ERROR] checkpoints do not exist')
            sys.exit()
 
        if self.use_cont_rot:
            n_dim_body=72+3
        else:
            n_dim_body=72

        self.model_h_latentD = 256
        self.model_h = HumanCVAES2(latentD_g=self.model_h_latentD,
                                     latentD_l=self.model_h_latentD,
                                     n_dim_body=n_dim_body,
                                     n_dim_scene=self.model_h_latentD,
                                     test=True)


        ### body mesh model
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')
        self.body_mesh_model = smplx.create(self.human_model_path, model_type='smplx',
                                       gender='neutral', ext='npz',
                                       num_pca_comps=12,
                                       create_global_orient=True,
                                       create_body_pose=True,
                                       create_betas=True,
                                       create_left_hand_pose=True,
                                       create_right_hand_pose=True,
                                       create_expression=True,
                                       create_jaw_pose=True,
                                       create_leye_pose=True,
                                       create_reye_pose=True,
                                       create_transl=True,
                                       batch_size=self.n_samples
                                       )


    def test(self, batch_gen):

        self.model_h.eval()
        self.model_h.to(self.device)

        self.vposer.to(self.device)
        self.body_mesh_model.to(self.device)

        ## load checkpoints
        ckp_list = sorted(glob.glob(os.path.join(self.ckpt_dir,'epoch-*.ckp')),
                            key=os.path.getmtime)
        ckp_path = ckp_list[-1]
        checkpoint = torch.load(ckp_path)
        print('[INFO] load checkpoints: ' + ckp_path)
        self.model_h.load_state_dict(checkpoint['model_h_state_dict'])

        ## get a batch of data for testing
        batch_gen.reset()
        
        test_data = batch_gen.next_batch(batch_size=1)

        depth_batch = test_data[0]
        seg_batch =   test_data[1]
        max_d_batch = test_data[2]
        cam_int_batch = test_data[3]
        cam_ext_batch = test_data[4]


        ## pass data to network
        xs = torch.cat([depth_batch, seg_batch],dim=1)

        xs_n = xs.repeat(self.n_samples, 1,1,1)
            
        noise_batch_g = torch.randn([self.n_samples, self.model_h_latentD], dtype=torch.float32,
                                            device=self.device)
        noise_batch_l = torch.randn([self.n_samples, self.model_h_latentD], dtype=torch.float32,
                                            device=self.device)

        if self.use_cont_rot:
            xhnr_gen = self.model_h.sample(xs_n, noise_batch_g, noise_batch_l)
            xhn_gen = GeometryTransformer.convert_to_3D_rot(xhnr_gen)
        else:
            xhnr_gen = self.model_h.sample(xs_n, noise_batch_g, noise_batch_l)

        xh_gen = GeometryTransformer.recover_global_T(xhn_gen, cam_int_batch, max_d_batch)

        body_param_list = BodyParamParser.body_params_encapsulate(xh_gen)
        
        scene_name = os.path.abspath(self.scene_file_path).split("/")[-2].split("_")[0]
        outdir = os.path.join(self.output_dir, scene_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        print('[INFO] save results to: '+outdir)
        for ii, body_param in enumerate(body_param_list):
            body_param['cam_ext'] = cam_ext_batch.detach().cpu().numpy()
            body_param['cam_int'] = cam_int_batch.detach().cpu().numpy()
            outfilename = os.path.join(outdir, 'body_gen_{:06d}.pkl'.format(ii+900))
            outfile = open(outfilename, 'wb')
            pickle.dump(body_param, outfile)
            outfile.close()


if __name__ == '__main__':

    proxe_path = '/is/cluster/yzhang/PROXE'
    test_file_list = ['MPH16_00157_01', 'N0SittingBooth_00162_01', 'MPH1Library_00034_01', 'N3OpenArea_00157_01']
    for fileid in range(len(test_file_list)):
        testconfig={
            'human_model_path': '/is/ps2/yzhang/body_models/VPoser',
            'scene_3d_path': os.path.join(proxe_path, 'scenes/'),
            'vposer_ckpt_path': '/is/ps2/yzhang/body_models/VPoser/vposer_v1_0',
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'ckpt_dir': 'checkpoints/checkpoints_virtualcams3_modelS2_batch32_epoch30_LRS0.0003_LRH0.0003_LossVposer0.001_LossKL0.1_LossContact0.001_LossCollision0.01',
            'test_data_path': os.path.join(proxe_path, 'snapshot/'+test_file_list[fileid]),
            'scene_file_path':os.path.join(proxe_path, 'snapshot/'+test_file_list[fileid]+'/rec_000000.mat'),
            'n_samples': 300,
            'use_cont_rot':True,
            'output_dir': 'results_proxe_stage2_sceneloss/virtualcams'
        }


        batch_gen = BatchGeneratorTest(dataset_path=testconfig['test_data_path'],
                                        device=testconfig['device'])
        test_op = TestOP(testconfig)

        test_op.test(batch_gen)

