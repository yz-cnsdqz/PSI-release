from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import sys, os, glob
import pdb
import json
import argparse
import numpy as np
import open3d as o3d

proj_path = '/is/ps2/yzhang/workspaces/PSI-internal'
sys.path.append(proj_path)


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
from batch_gen_hdf5 import BatchGeneratorWithSceneMesh

import pdb

import time




class TrainOP:
    def __init__(self, trainconfig, lossconfig):
        for key, val in trainconfig.items():
            setattr(self, key, val)


        for key, val in lossconfig.items():
            setattr(self, key, val)


        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


        ### define model

        if self.use_cont_rot:
            n_dim_body=72+3
        else:
            n_dim_body=72

        self.model_h_latentD = 256
        self.model_h = HumanCVAES2(latentD_g=self.model_h_latentD,
                                     latentD_l=self.model_h_latentD,
                                     scene_model_ckpt=self.scene_model_ckpt,
                                     n_dim_body=n_dim_body,
                                     n_dim_scene=self.model_h_latentD)

        self.optimizer_h = optim.Adam(self.model_h.parameters(), 
                                      lr=self.init_lr_h)


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
                                            batch_size=self.batch_size
                                            )

        self.smplx_face_idx = np.load(os.path.join(self.human_model_path, 
                                        'smplx/SMPLX_NEUTRAL.npz'),
                                    allow_pickle=True)['f'].reshape(-1,3)
        self.smplx_face_idx = torch.tensor(self.smplx_face_idx.astype(np.int64), 
                                            device=self.device)

        print('--[INFO] device: '+str(torch.cuda.get_device_name(self.device)) )






    def cal_loss(self, xs, xh, eps_g, eps_l, cam_ext, cam_int, max_d,
                 scene_verts, scene_face,
                 s_grid_min_batch, s_grid_max_batch, s_grid_sdf_batch,
                 ep):



        # normalize global trans
        xhn = GeometryTransformer.normalize_global_T(xh, cam_int, max_d)

        # convert global rotation
        xhnr = GeometryTransformer.convert_to_6D_rot(xhn)
        [xhnr_rec, mu_g, logsigma2_g, 
            mu_l, logsigma2_l] = self.model_h(xhnr, eps_g, eps_l, xs)
        xhn_rec = GeometryTransformer.convert_to_3D_rot(xhnr_rec)


        # recover global trans
        xh_rec = GeometryTransformer.recover_global_T(xhn_rec, cam_int, max_d)

        loss_rec_t = self.weight_loss_rec_h*( 0.5*F.l1_loss(xhnr_rec[:,:3], xhnr[:,:3])
                                             +0.5*F.l1_loss(xh_rec[:,:3], xh[:,:3]))
        loss_rec_p = self.weight_loss_rec_h*F.l1_loss(xhnr_rec[:,3:], xhnr[:,3:])


        ### kl divergence loss
        fca = 1.0
        if self.loss_weight_anealing:
            fca = min(1.0, max(float(ep) / (self.epoch*0.75),0) )

        loss_KL_g = fca**2  *self.weight_loss_kl * 0.5*torch.mean(torch.exp(logsigma2_g) +mu_g**2 -1.0 -logsigma2_g)
        loss_KL_l = fca**2  *self.weight_loss_kl * 0.5*torch.mean(torch.exp(logsigma2_l) +mu_l**2 -1.0 -logsigma2_l)



        ### Vposer loss
        vposer_pose = xh_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)


        ### contact loss
        ## (1) get the reconstructed body mesh
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

        ## (2) select contact vertex according to prox annotation
        vid, fid = GeometryTransformer.get_contact_id(body_segments_folder=self.contact_id_folder, 
                                  contact_body_parts=self.contact_part)

        body_verts_contact_batch = body_verts_batch[:, vid, :]


        ## (3) compute chamfer loss between pcd_batch and body_verts_batch
        dist_chamfer_contact = ext.chamferDist()
        contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch.contiguous(),
                                                        scene_verts.contiguous())

        fcc = 0.0
        if ep > 0.75*self.epoch:
            fcc = 1.0

        loss_contact = fcc *self.weight_contact * torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0)  )


        ### SDF scene penetration loss
        s_grid_min_batch = s_grid_min_batch.unsqueeze(1)
        s_grid_max_batch = s_grid_max_batch.unsqueeze(1)

        norm_verts_batch = (body_verts_batch - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) *2 -1
        n_verts = norm_verts_batch.shape[1]
        body_sdf_batch = F.grid_sample(s_grid_sdf_batch.unsqueeze(1),
                                        norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
                                        padding_mode='border')


        # if there are no penetrating vertices then set sdf_penetration_loss = 0
        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            loss_sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()

        fsp = 0.0
        if ep > 0.75*self.epoch:
            fsp = 1.0

        loss_sdf_pene = fsp*self.weight_collision*loss_sdf_pene

        return loss_rec_t, loss_rec_p, loss_KL_g, loss_KL_l, loss_contact, loss_vposer, loss_sdf_pene





    def train(self, batch_gen):

        self.model_h.train()
        self.model_h.to(self.device)

        self.vposer.to(self.device)
        self.body_mesh_model.to(self.device)


        starting_ep = 0
        if self.resume_training:
            ckp_list = sorted(glob.glob(os.path.join(self.save_dir,'epoch-*.ckp')),
                                key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])

                self.model_h.load_state_dict(checkpoint['model_h_state_dict'])
                self.optimizer_h.load_state_dict(checkpoint['optimizer_h_state_dict'])

                starting_ep = checkpoint['epoch']
                print('[INFO] --resuming training from {}'.format(ckp_list[-1]))


        print('--[INFO] start training')
        start_time = time.time()
        for ep in range(starting_ep, self.epoch):
            epoch_loss_rec_h = 0
            epoch_loss_KL = 0
            epoch_loss_vposer = 0
            epoch_loss_contact = 0
            epoch_loss_collision = 0

            while batch_gen.has_next_batch():

                self.optimizer_h.zero_grad()

                ######### get training data batch #########
                train_data = batch_gen.next_batch(self.batch_size)

                if train_data is None:
                    continue


                [depth_batch, seg_batch, body_batch,
                cam_ext_batch, cam_int_batch,max_d_batch,
                s_verts_batch, s_faces_batch,
                s_grid_min_batch, s_grid_max_batch,
                s_grid_dim_batch, s_grid_sdf_batch] = train_data



                ### get noise
                noise_batch_l = torch.randn([self.batch_size, self.model_h_latentD], dtype=torch.float32,
                                            device=self.device)

                noise_batch_g = torch.randn([self.batch_size, self.model_h_latentD], dtype=torch.float32,
                                            device=self.device)


                ### calculate loss
                [loss_rec_t,
                 loss_rec_p,
                 loss_KL_g,
                 loss_KL_l,
                 loss_contact,
                 loss_vposer,
                 loss_sdf_pene] =  self.cal_loss(xs=torch.cat([depth_batch,seg_batch],dim=1),
                                               xh=body_batch,
                                               eps_g=noise_batch_g,
                                               eps_l=noise_batch_l,
                                               cam_ext=cam_ext_batch,
                                               cam_int = cam_int_batch,
                                               max_d = max_d_batch,
                                               scene_verts=s_verts_batch,
                                               scene_face=s_faces_batch,
                                               s_grid_min_batch=s_grid_min_batch,
                                               s_grid_max_batch=s_grid_max_batch,
                                               s_grid_sdf_batch=s_grid_sdf_batch,
                                               ep=ep)


                loss_coll = loss_sdf_pene
                loss_h = loss_rec_t+loss_rec_p + loss_vposer+loss_KL_g + loss_KL_l + loss_contact+loss_coll


                loss_h.backward(retain_graph=True)
                self.optimizer_h.step()


                if self.verbose:
                    print("---in [epoch {:d}]: rec_t={:f}, rec_p={:f}, kl_g={:f}, kl_l={:f}, vp={:f}, contact={:f}, collision={:f}"
                            .format(ep + 1,
                                    loss_rec_t.item(),
                                    loss_rec_p.item(),
                                    loss_KL_g.item(),
                                    loss_KL_l.item(),
                                    loss_vposer.item(),
                                    loss_contact.item(),
                                    loss_coll.item() ))


                elapsed_time = (time.time() - start_time)/3600.0
                if elapsed_time >= 2:
                    start_time = time.time()
                    torch.save({
                                'epoch': ep+1,
                                'model_h_state_dict': self.model_h.state_dict(),
                                'optimizer_h_state_dict': self.optimizer_h.state_dict(),
                                }, self.save_dir + "/epoch-{:06d}".format(ep + 1) + ".ckp")

            batch_gen.reset()


            ## save checkpoints
            if (ep+1) % 10 == 0:
                torch.save({
                            'epoch': ep+1,
                            'model_h_state_dict': self.model_h.state_dict(),
                            'optimizer_h_state_dict': self.optimizer_h.state_dict(),
                            }, self.save_dir + "/epoch-{:06d}".format(ep + 1) + ".ckp")


            # if self.verbose:
            #     print("--[epoch {:d}]: rec_s={:f}, rec_h={:f}, kl={:f}, vp={:f}, contact={:f}, collision={:f}"
            #                 .format(ep + 1,
            #                         epoch_loss_rec_s / (len(batch_gen.rec_list)//self.batch_size),
            #                         epoch_loss_rec_h / (len(batch_gen.rec_list)//self.batch_size),
            #                         epoch_loss_KL / (len(batch_gen.rec_list)//self.batch_size),
            #                         epoch_loss_vposer / (len(batch_gen.rec_list)//self.batch_size),
            #                         epoch_loss_contact / (len(batch_gen.rec_list)//self.batch_size),
            #                         epoch_loss_collision / (len(batch_gen.rec_list)//self.batch_size) ))



        if self.verbose:
            print('[INFO]: Training completes!')
            print()






if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),
                        help='dir for checkpoints')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size to train')

    parser.add_argument('--lr_s', type=float, default=0.001,
                        help='initial learing rate')

    parser.add_argument('--lr_h', type=float, default=0.0001,
                        help='initial learing rate')

    parser.add_argument('--num_epoch', type=int, default=50,
                        help='#epochs to train')

    parser.add_argument('--weight_loss_vposer', type=float, default=1e-3)
    parser.add_argument('--weight_loss_kl', type=float, default=1e-1)
    parser.add_argument('--weight_loss_contact', type=float, default=1e-1)
    parser.add_argument('--weight_loss_collision', type=float, default=1e-1)
    parser.add_argument('--only_vircam', type=int, default=0)
    parser.add_argument('--use_all', type=int, default=0)


    args = parser.parse_args()
    save_dir = args.save_dir


    ### setup dataset paths and training configs.
    dataset_path = '/is/cluster/yzhang/PROXE'


    if save_dir == 'None':
        print('[error] the checkpoint save directory should be specified.')
        sys.exit(0)
    else:
        save_dir = save_dir
        resume_training=True

    if args.only_vircam == 1:
        trainfile = os.path.join(dataset_path, 'virtualcams_v2.hdf5')
    else:
        trainfile = [os.path.join(dataset_path, 'virtualcams_v2.hdf5'), 
                     os.path.join(dataset_path, 'realcams_v2.hdf5')]



    trainconfig={
        'train_data_path': trainfile,
        'scene_verts_path': os.path.join(dataset_path, 'scenes_downsampled'),
        'scene_sdf_path': os.path.join(dataset_path,'scenes_sdf'),
        'scene_model_ckpt': os.path.join(proj_path,'data/resnet18.pth'),
        'human_model_path': '/is/ps2/yzhang/body_models/VPoser',
        'vposer_ckpt_path': '/is/ps2/yzhang/body_models/VPoser/vposer_v1_0',
        'init_lr_s': args.lr_s,
        'init_lr_h': args.lr_h,
        'batch_size': args.batch_size, # >1
        'epoch': args.num_epoch,
        'loss_weight_anealing': True,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'fine_tuning': None, # or specify the path to resume training
        'save_dir': save_dir, #'/is/ps2/yzhang/workspaces/smpl-env-gen-3d/checkpoints',
        'contact_id_folder': os.path.join(dataset_path, 'body_segments'),
        'contact_part': ['back','butt','L_Hand','R_Hand','L_Leg','R_Leg','thighs'],
        'saving_per_X_ep': 2,
        'verbose': True,
        'use_cont_rot': True,
        'resume_training': resume_training
    }



    lossconfig={
        'weight_loss_rec_s': 1.0,
        'weight_loss_rec_h': 1.0,
        'weight_loss_vposer':args.weight_loss_vposer,
        'weight_loss_kl': args.weight_loss_kl,
        'weight_contact': args.weight_loss_contact,
        'weight_collision' : args.weight_loss_collision
    }

    if args.use_all == 1:
        mode='all'
    else:
        mode='train'


    batch_gen = BatchGeneratorWithSceneMesh(dataset_path=trainconfig['train_data_path'],
                                            scene_verts_path = trainconfig['scene_verts_path'],
                                            scene_sdf_path = trainconfig['scene_sdf_path'],
                                            mode=mode,
                                            device=trainconfig['device'],
                                            read_all_to_ram=True)



    train_op = TrainOP(trainconfig, lossconfig)
    train_op.train(batch_gen)

