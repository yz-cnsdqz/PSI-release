from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import open3d as o3d
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

import random
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import sys, os
import json

from net_layers import BodyGlobalPoseVAE, BodyLocalPoseVAE, ResBlock




################################################################################
## Conditional VAE of the human body, 
## Input: 72/75-dim, [T (3d vector), R (3d/6d), shape (10d), pose (32d), 
#         lefthand (12d), righthand (12d)]
## Note that, it requires pre-trained VPoser and latent variable of scene
################################################################################


class ContinousRotReprDecoder(nn.Module):
    '''
    - this class encodes/decodes rotations with the 6D continuous representation
    - Zhou et al., On the continuity of rotation representations in neural networks
    - also used in the VPoser (see smplx)
    '''

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot







class GeometryTransformer():
    
    @staticmethod
    def get_contact_id(body_segments_folder, contact_body_parts=['L_Hand', 'R_Hand']):

        contact_verts_ids = []
        contact_faces_ids = []

        for part in contact_body_parts:
            with open(os.path.join(body_segments_folder, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
                contact_faces_ids.append(list(set(data["faces_ind"])))

        contact_verts_ids = np.concatenate(contact_verts_ids)
        contact_faces_ids = np.concatenate(contact_faces_ids)


        return contact_verts_ids, contact_faces_ids

    @staticmethod
    def convert_to_6D_rot(x_batch):
        xt = x_batch[:,:3]
        xr = x_batch[:,3:6]
        xb = x_batch[:, 6:]

        xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
        xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

        return torch.cat([xt, xr_repr, xb], dim=-1)

    @staticmethod
    def convert_to_3D_rot(x_batch):
        xt = x_batch[:,:3]
        xr = x_batch[:,3:9]
        xb = x_batch[:,9:]

        xr_mat = ContinousRotReprDecoder.decode(xr) # return [:,3,3]
        xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

        return torch.cat([xt, xr_aa, xb], dim=-1)



    @staticmethod
    def verts_transform(verts_batch, cam_ext_batch):
        verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
        verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                    cam_ext_batch.permute(0,2,1))

        verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
        
        return verts_batch_transformed


    @staticmethod
    def recover_global_T(x_batch, cam_intrisic, max_depth):
        xt_batch = x_batch[:,:3]
        xr_batch = x_batch[:,3:]

        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        # fx_batch = 1000
        # fy_batch = 1000
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        
        z = (xt_batch[:, 2]+1.0)/2.0 * max_depth

        x = xt_batch[:,0] * z / s_ / fx_batch
        y = xt_batch[:,1] * z / s_ / fy_batch
        
        xt_batch_recoverd = torch.stack([x,y,z],dim=-1)

        return torch.cat([xt_batch_recoverd, xr_batch],dim=-1)


    @staticmethod
    def normalize_global_T(x_batch, cam_intrisic, max_depth):
        '''
        according to the camera intrisics and maximal depth,
        normalize the global translate to [-1, 1] for X, Y and Z.
        input: [transl, rotation, local params]
        '''

        xt_batch = x_batch[:,:3]
        xr_batch = x_batch[:,3:]

        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        x = s_* xt_batch[:,0]*fx_batch / (xt_batch[:,2] + 1e-6)
        y = s_* xt_batch[:,1]*fy_batch / (xt_batch[:,2] + 1e-6)

        z = 2.0*xt_batch[:,2] / max_depth - 1.0

        xt_batch_normalized = torch.stack([x,y,z],dim=-1)


        return torch.cat([xt_batch_normalized, xr_batch],dim=-1)

















class BodyParamParser():

    @staticmethod
    def body_params_encapsulate(x_body_rec):
        x_body_rec_np = x_body_rec.detach().cpu().numpy()
        n_batch = x_body_rec_np.shape[0]
        rec_list = []

        for b in range(n_batch):
            body_params_batch_rec={}
            body_params_batch_rec['transl'] = x_body_rec_np[b:b+1,:3]
            body_params_batch_rec['global_orient'] = x_body_rec_np[b:b+1,3:6]
            body_params_batch_rec['betas'] = x_body_rec_np[b:b+1,6:16]
            body_params_batch_rec['body_pose'] = x_body_rec_np[b:b+1,16:48]
            body_params_batch_rec['left_hand_pose'] = x_body_rec_np[b:b+1,48:60]
            body_params_batch_rec['right_hand_pose'] = x_body_rec_np[b:b+1,60:]
            rec_list.append(body_params_batch_rec)

        return rec_list


    @staticmethod
    def body_params_encapsulate_batch(x_body_rec):
        
        body_params_batch_rec={}
        body_params_batch_rec['transl'] = x_body_rec[:,:3]
        body_params_batch_rec['global_orient'] = x_body_rec[:,3:6]
        body_params_batch_rec['betas'] = x_body_rec[:,6:16]
        body_params_batch_rec['body_pose_vp'] = x_body_rec[:,16:48]
        body_params_batch_rec['left_hand_pose'] = x_body_rec[:,48:60]
        body_params_batch_rec['right_hand_pose'] = x_body_rec[:,60:]

        return body_params_batch_rec

    @staticmethod
    def body_params_encapsulate_latent(x_body_rec, eps=None):

        x_body_rec_np = x_body_rec.detach().cpu().numpy()
        eps_np = eps.detach().cpu().numpy()

        n_batch = x_body_rec_np.shape[0]
        rec_list = []

        for b in range(n_batch):
            body_params_batch_rec={}
            body_params_batch_rec['transl'] = x_body_rec_np[b:b+1,:3]
            body_params_batch_rec['global_orient'] = x_body_rec_np[b:b+1,3:6]
            body_params_batch_rec['betas'] = x_body_rec_np[b:b+1,6:16]
            body_params_batch_rec['body_pose'] = x_body_rec_np[b:b+1,16:48]
            body_params_batch_rec['left_hand_pose'] = x_body_rec_np[b:b+1,48:60]
            body_params_batch_rec['right_hand_pose'] = x_body_rec_np[b:b+1,60:]
            body_params_batch_rec['z'] = eps_np[b:b+1, :]
            rec_list.append(body_params_batch_rec)

        return rec_list

    @staticmethod
    def body_params_parse(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']

        x_body = np.concatenate([x_body_T, x_body_R, 
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu


    @staticmethod
    def body_params_parse_fitting(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']
        cam_ext = torch.tensor(body_params_batch['cam_ext'], dtype=torch.float32).cuda()
        cam_int = torch.tensor(body_params_batch['cam_int'], dtype=torch.float32).cuda()
        
        x_body = np.concatenate([x_body_T, x_body_R, 
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu, cam_ext, cam_int






class HumanCVAES2(nn.Module):
    '''
    the two-stage conditional VAE
    '''
    def __init__(self, 
                 latentD_g=512,
                 latentD_l=512,
                 scene_model_ckpt=None,
                 n_dim_body=72,
                 n_dim_scene=128,
                 test=False
                 ):
        super(HumanCVAES2, self).__init__()

        self.latentD_g = latentD_g
        self.latentD_l = latentD_l
        self.n_dim_g = 3  
        self.n_dim_l = n_dim_body-self.n_dim_g

        self.trans_vae=BodyGlobalPoseVAE(zdim=32,in_dim=2,num_hidden=latentD_g,  
                    pretrained_resnet=scene_model_ckpt,
                    test=test)
        self.pose_vae = BodyLocalPoseVAE(zdim=32,in_dim=2,num_hidden=latentD_g,
                    pretrained_resnet=scene_model_ckpt,
                    test=test)



    def forward(self, x_body, eps_g, eps_l, x_s):
        '''
        input: x_body: body representation, [batch, 72D/75D]
               z_s: scene representation, [batch, 128D]

        '''
        b_ = x_s.shape[0]
        x_g = x_body[:, :3]
        x_l = x_body[:, 3:]
        
        x_g_rec, mu_g, logsigma2_g = self.trans_vae(x_s, x_g)
        x_l_rec, mu_l, logsigma2_l = self.pose_vae(x_s, x_g_rec, x_l)

        x_body_rec = torch.cat([x_g_rec, x_l_rec], dim=1)


        return x_body_rec, mu_g, logsigma2_g, mu_l, logsigma2_l

    


    def sample(self, x_s, eps_g=None, eps_l=None):

        b_ = x_s.shape[0]

        x_g_gen = self.trans_vae(x_s)
        x_l_gen = self.pose_vae(x_s, x_g_gen)


        x_body_gen = torch.cat([x_g_gen, x_l_gen],dim=1)

        return x_body_gen










class HumanCVAES1(nn.Module):
    ''' 
    one stage-model, sampling global trans and local pose at the same time
    '''
    def __init__(self, 
                 latentD=512, 
                 n_dim_body=75,
                 scene_model_ckpt=None,
                 test=False
                 ):
        super(HumanCVAES1, self).__init__()

        self.test = test
        self.eps_d = 32
        
        ## scene encoder
        resnet = torchvision.models.resnet18()
        if scene_model_ckpt is not None:
            print('[INFO][SceneNet] Using pretrained resnet18 weights.')
            resnet.load_state_dict(torch.load(scene_model_ckpt))
        removed = list(resnet.children())[1:6]
        self.resnet = nn.Sequential(nn.Conv2d(2, 64, kernel_size=7, 
                                                stride=2, padding=3,
                                                bias=False),
                                    *removed)
        self.conv = nn.Conv2d(128,32,3,1,1) # b x f_dim x 16 x 16
        self.fc = nn.Linear(32*16*16,latentD)




        ## human encoder
        self.linear_in = nn.Linear(n_dim_body, latentD)
        self.human_encoder = nn.Sequential(ResBlock(2*latentD),
                                           ResBlock(2*latentD))

        self.mu_enc = nn.Linear(2*latentD, self.eps_d)
        self.logvar_enc = nn.Linear(2*latentD, self.eps_d)


        self.linear_latent = nn.Linear(self.eps_d, latentD)
        self.human_decoder = nn.Sequential(ResBlock(2*latentD),
                                           ResBlock(2*latentD))

        self.linear_out = nn.Linear(2*latentD, n_dim_body)



    def _sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        return eps.mul(var).add_(mu)


    def forward(self, x_body, x_s):
        '''
        input: x_body: body representation, [batch, 72D/75D]
               z_s: scene representation, [batch, 128D]

        '''
        
        b_ = x_s.shape[0]
        z_s = self.conv(self.resnet(x_s))
        z_s = self.fc(z_s.view(b_, -1))

        z_h = self.linear_in(x_body)


        z_hs = torch.cat([z_h, z_s], dim=1)
        z_hs = self.human_encoder(z_hs)

        mu = self.mu_enc(z_hs)
        logvar = self.logvar_enc(z_hs)

        z_h = self._sampler(mu, logvar)
        z_h = self.linear_latent(z_h)
        z_hs = torch.cat([z_h, z_s], dim=1)

        z_hs = self.human_decoder(z_hs)

        x_body_rec = self.linear_out(z_hs)


        return x_body_rec, mu, logvar

    

    def sample(self, x_s, **kwargs):

        b_ = x_s.shape[0]
        z_s = self.conv(self.resnet(x_s))
        z_s = self.fc(z_s.view(b_, -1))

        eps = torch.randn([b_, self.eps_d],dtype=torch.float32).cuda()
        z_h = self.linear_latent(eps)
        z_hs = torch.cat([z_h, z_s], dim=1)
        z_hs = self.human_decoder(z_hs)
        x_body_gen = self.linear_out(z_hs)


        return x_body_gen



    def sample_line(self, x_s, **kwargs):

        b_ = x_s.shape[0]
        z_s = self.conv(self.resnet(x_s))
        z_s = self.fc(z_s.view(b_, -1))


        eps_np = np.repeat(np.expand_dims(np.arange(-3,3,6.0/b_), axis=1), 
                           self.eps_d, axis=1)

        eps = torch.tensor(eps_np, dtype=torch.float32).cuda()

        z_h = self.linear_latent(eps)
        z_hs = torch.cat([z_h, z_s], dim=1)
        z_hs = self.human_decoder(z_hs)
        x_body_gen = self.linear_out(z_hs)


        return x_body_gen, eps











