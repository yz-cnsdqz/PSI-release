'''
this is not the original prox fitting, but a modified version just for post-processing
our generated results. The input is the smplx body parameters, and the optimization 
is based on the scene sdf and the contact loss
'''


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







class FittingOP:
    def __init__(self, fittingconfig, lossconfig):


        for key, val in fittingconfig.items():
            setattr(self, key, val)


        for key, val in lossconfig.items():
            setattr(self, key, val)


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
                                       batch_size=self.batch_size
                                       )
        self.vposer.to(self.device)
        self.body_mesh_model.to(self.device)

        self.xhr_rec = Variable(torch.randn(1,75).to(self.device), requires_grad=True)
        self.optimizer = optim.Adam([self.xhr_rec], lr=self.init_lr_h)




        ## read scene sdf
        with open(self.scene_sdf_path+'.json') as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'])
                grid_max = np.array(sdf_data['max'])
                grid_dim = sdf_data['dim']
        sdf = np.load(self.scene_sdf_path + '_sdf.npy').reshape(grid_dim, grid_dim, grid_dim)

        self.s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=self.device).unsqueeze(0)


        ## read scene vertices
        scene_o3d = o3d.io.read_triangle_mesh(self.scene_verts_path)
        scene_verts = np.asarray(scene_o3d.vertices)
        self.s_verts_batch = torch.tensor(scene_verts, dtype=torch.float32, device=self.device).unsqueeze(0)





    def cal_loss(self, xhr, cam_ext):
        
        ### reconstruction loss
        loss_rec = self.weight_loss_rec*F.l1_loss(xhr, self.xhr_rec)
        xh_rec = GeometryTransformer.convert_to_3D_rot(self.xhr_rec)

        ### vposer loss
        vposer_pose = xh_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)


        ### contact loss
        body_param_rec = BodyParamParser.body_params_encapsulate_batch(xh_rec)
        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)
 
        body_param_ = {}
        for key in body_param_rec.keys():
            if key in ['body_pose_vp']:
                continue
            else:
                body_param_[key] = body_param_rec[key]

        smplx_output = self.body_mesh_model(return_verts=True, 
                                              body_pose=joint_rot_batch,
                                              **body_param_)
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch = GeometryTransformer.verts_transform(body_verts_batch, cam_ext)

        vid, fid = GeometryTransformer.get_contact_id(
                                body_segments_folder=self.contact_id_folder,
                                contact_body_parts=self.contact_part)
        body_verts_contact_batch = body_verts_batch[:, vid, :]

        dist_chamfer_contact = ext.chamferDist()
        contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch.contiguous(), 
                                                self.s_verts_batch.contiguous())

        loss_contact = self.weight_contact * torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0))  


        ### sdf collision loss
        s_grid_min_batch = self.s_grid_min_batch.unsqueeze(1)
        s_grid_max_batch = self.s_grid_max_batch.unsqueeze(1)

        norm_verts_batch = (body_verts_batch - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) *2 -1
        n_verts = norm_verts_batch.shape[1]
        body_sdf_batch = F.grid_sample(self.s_sdf_batch.unsqueeze(1), 
                                        norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
                                        padding_mode='border')


        # if there are no penetrating vertices then set sdf_penetration_loss = 0
        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            loss_sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()

        loss_collision = self.weight_collision*loss_sdf_pene


        return loss_rec, loss_vposer, loss_contact, loss_collision




    def fitting(self, input_data_file):


        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)

        xh, self.cam_ext, self.cam_int= BodyParamParser.body_params_parse_fitting(body_param_input)
        xhr = GeometryTransformer.convert_to_6D_rot(xh)
        self.xhr_rec.data = xhr.clone()

        T_mat = np.eye(4)
        T_mat[1,:] = np.array([0,-1,0,0])
        T_mat[2,:] = np.array([0,0,-1,0])
        T_mat = torch.tensor(T_mat, dtype=torch.float32, device=self.device)
        T_mat = T_mat.unsqueeze(0)
        trans = torch.matmul(self.cam_ext[:1],T_mat)
        


        for ii in range(self.num_iter):

            self.optimizer.zero_grad()

            loss_rec, loss_vposer, loss_contact, loss_collision = self.cal_loss(xhr, trans)
            loss = loss_rec + loss_vposer + loss_contact + loss_collision
            if self.verbose:
                print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, l_contact={:f}, l_collision={:f}'.format(
                                        ii, loss_rec.item(), loss_vposer.item(), 
                                        loss_contact.item(), loss_collision.item()) )

            loss.backward(retain_graph=True)
            self.optimizer.step()


        print('[INFO][fitting] fitting finish, returning optimal value')


        xh_rec =  GeometryTransformer.convert_to_3D_rot(self.xhr_rec)

        return xh_rec



    def save_result(self, xh_rec, output_data_file):

        dirname = os.path.dirname(output_data_file)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        body_param_list = BodyParamParser.body_params_encapsulate(xh_rec)
        print('[INFO] save results to: '+output_data_file)
        for ii, body_param in enumerate(body_param_list):
            # print(body_param['transl'])
            # print(body_param['global_orient'])
            # print()
            body_param['cam_ext'] = self.cam_ext.detach().cpu().numpy()
            body_param['cam_int'] = self.cam_int.detach().cpu().numpy()
            outfile = open(output_data_file, 'wb')
            pickle.dump(body_param, outfile)
            outfile.close()


if __name__=='__main__':


    gen_path = sys.argv[1]
    fit_path = sys.argv[2]

    scene_test_list = ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge', 
                        '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1', 
                        'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0', 
                        'zsNo4HB9uLZ-livingroom0_13']
    
    mp3dr_path = '/is/cluster/yzhang/mp3d-rooms'

    for scenename in scene_test_list:

        fittingconfig={
            'scene_verts_path': os.path.join(mp3dr_path, scenename+'.ply'),
            'scene_sdf_path': os.path.join(mp3dr_path, 'sdf/'+scenename),
            'human_model_path': '/is/ps2/yzhang/body_models/VPoser',
            'vposer_ckpt_path': '/is/ps2/yzhang/body_models/VPoser/vposer_v1_0',
            'init_lr_h': 0.1,
            'num_iter': 50,
            'batch_size': 1, 
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'contact_id_folder': '/is/cluster/work/yzhang/PROX/body_segments',
            'contact_part': ['back','butt','L_Hand','R_Hand','L_Leg','R_Leg','thighs'],
            'verbose': False
        }

        lossconfig={
            'weight_loss_rec': 1,
            'weight_loss_vposer':0.01,
            'weight_contact': 0.1,
            'weight_collision' : 0.5
        }


        fop = FittingOP(fittingconfig, lossconfig)



        for ii in range(10000):
            input_data_file = os.path.join(gen_path,scenename+'/body_gen_{:06d}.pkl'.format(ii))

            if not os.path.exists(input_data_file):
                continue

            output_data_file = os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii))
            if os.path.exists(output_data_file):
                continue

            
            xh_rec = fop.fitting(input_data_file)

            if xh_rec is None:
                continue
            fop.save_result(xh_rec, output_data_file)

