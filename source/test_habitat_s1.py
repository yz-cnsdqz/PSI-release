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

from cvae import BodyParamParser, HumanCVAES1, GeometryTransformer








class TestOP:
    def __init__(self, testconfig):
        for key, val in testconfig.items():
            setattr(self, key, val)


        if not os.path.exists(self.ckpt_dir):
            print('--[ERROR] checkpoints do not exist')
            sys.exit()

        #define model
 
        if self.use_cont_rot:
            n_dim_body=72+3
        else:
            n_dim_body=72

        self.model_h_latentD = 256
        self.model_h = HumanCVAES1(latentD=self.model_h_latentD,
                                  n_dim_body=n_dim_body)



        ### body mesh model
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')
        self.body_mesh_model = smplx.create(self.human_model_path, 
                                            model_type='smplx',
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


    def data_preprocessing(self, img, modality, target_domain_size=[128, 128], 
                                filename=None):

        """
        input:
            - img (depthmap or semantic map): [height, width].
            - modality: 'depth' or 'seg'
        output:
            canvas: with shape of target_domain_size, where the input is in the
                    center tightly, with shape target_domain_size
            factor: the resizing factor
        """

        # prepare the canvas
        img_shape_o = img.shape
        canvas = torch.zeros([1,1]+target_domain_size, dtype=torch.float32,
                             device=self.device)


        # filter out unavailable values
        if modality == 'depth':
            img[img>6.0]=6.0


        if modality == 'seg':
            img[img>41] = 41

        max_val = torch.max(img)
        _img = 2* img / max_val - 1.0

        # put _img to the canvas
        if img_shape_o[0]>= img_shape_o[1]:
            factor = float(target_domain_size[0]) / img_shape_o[0]
            target_height = target_domain_size[0]
            target_width = int(img_shape_o[1] * factor) //2 *2 

            # for depth map we use bilinear interpolation in resizing
            # for segmentation map we use bilinear interpolation as well.
            # note that float semantic label is not real in practice, but
            # helpful in our work
            target_size = [target_height, target_width]
            
            _img = _img.view(1,1,img_shape_o[0],img_shape_o[1])
            img_resize = F.interpolate(_img, size=target_size, mode='bilinear',
                                        align_corners=False)
            
            na = target_width
            nb = target_domain_size[1]
            lower = (nb //2) - (na //2)
            upper = (nb //2) + (na //2)

            canvas[:,:,:, lower:upper] = img_resize


        else:
            factor = float(target_domain_size[1]) / img_shape_o[1]

            target_height = int(factor*img_shape_o[0]) //2 *2
            target_width = target_domain_size[1]

            target_size = [target_height, target_width]
            _img = _img.view(1,1,img_shape_o[0],img_shape_o[1])
            img_resize = F.interpolate(_img, size=target_size, mode='bilinear',
                                        align_corners=False)

            na = target_height
            nb = target_domain_size[0]
            lower = (nb //2) - (na //2)
            upper = (nb //2) + (na //2)

            canvas[:,:,lower:upper, :] = img_resize

        return canvas, factor, max_val





    def test(self):

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


        ## read data from here!
        cam_file_list = sorted(glob.glob(self.test_data_path+'/cam_*'))

        for ii, cam_file in enumerate(cam_file_list):

            cam_params = np.load(cam_file, allow_pickle=True, encoding='latin1').item()

            depth0 = torch.tensor(np.load(cam_file.replace('cam','depth')), dtype=torch.float32, device=self.device)
            seg0 = torch.tensor(np.load(cam_file.replace('cam','seg')),dtype=torch.float32, device=self.device)


            cam_ext = torch.tensor(cam_params['cam_ext'], dtype=torch.float32, device=self.device).unsqueeze(0) #[1,4,4]
            cam_int = torch.tensor(cam_params['cam_int'], dtype=torch.float32, device=self.device).unsqueeze(0) # [1,3,3]

            depth, _, max_d = self.data_preprocessing(depth0, 'depth', target_domain_size=[128, 128]) #[1,1,128,128]
            max_d = max_d.view(1)
            seg, _, _ = self.data_preprocessing(seg0, 'depth', target_domain_size=[128, 128])


            xs = torch.cat([depth, seg], dim=1)
            xs_batch = xs.repeat(self.n_samples, 1,1,1)
            max_d_batch = max_d.repeat(self.n_samples)
            cam_int_batch = cam_int.repeat(self.n_samples, 1,1)
            cam_ext_batch = cam_ext.repeat(self.n_samples, 1,1)


            xhnr_gen = self.model_h.sample(xs_batch)
            xhn_gen = GeometryTransformer.convert_to_3D_rot(xhnr_gen)            
            xh_gen = GeometryTransformer.recover_global_T(xhn_gen, cam_int_batch, max_d_batch)


            body_param_list = BodyParamParser.body_params_encapsulate(xh_gen)

            

            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)


            print('[INFO] save results to: '+self.outdir)
            for jj, body_param in enumerate(body_param_list):
                body_param['cam_ext'] = cam_ext_batch.detach().cpu().numpy()
                body_param['cam_int'] = cam_int_batch.detach().cpu().numpy()
                outfilename = os.path.join(self.outdir, 'body_gen_{:06d}.pkl'.format(self.n_samples*ii+jj))
                outfile = open(outfilename, 'wb')
                pickle.dump(body_param, outfile)
                outfile.close()



if __name__ == '__main__':

    scene_list = ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge', 
                  '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1', 
                  'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0', 
                  'zsNo4HB9uLZ-livingroom0_13']

    mp3dr_path = '/is/cluster/yzhang/mp3d-rooms'

    for fileid in range(len(scene_list)):
        print('[INFO] processing: '+scene_list[fileid])
        testconfig={
            'outdir': 'results_habitat_stage1_sceneloss/virtualcams/'+scene_list[fileid],
            'ckpt_dir': 'checkpoints_prox_modelS1_batch32_epoch30_LRS0.0003_LRH0.0003_LossVposer0.001_LossKL0.1_LossContact0.001_LossCollision0.01',
            'human_model_path': '/is/ps2/yzhang/body_models/VPoser',
            'vposer_ckpt_path': '/is/ps2/yzhang/body_models/VPoser/vposer_v1_0',
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'test_data_path': os.path.join(mp3dr_path ,scene_list[fileid]+'-sensor'),
            'n_samples': 200,
            'use_cont_rot':True
        }


        test_op = TestOP(testconfig)


        test_op.test()

