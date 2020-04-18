from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import open3d as o3d
import json
from scipy.spatial.distance import cdist
import os, sys, glob
import scipy.io as sio
import random
import numpy as np
import h5py
import torch
from torch.utils import data
import torch.nn.functional as F

import pdb

## proxe
### scenes for training: 
# ['BasementSittingBooth', 'MPH8', 'MPH11',
# 'MPH112', 'N0Sofa', 'N3Library', 'N3Office', 'Werkraum'
# ]


### scenes for testing: 
# ['MPH16', 'MPH1Library','N0SittingBooth', 'N3OpenArea']





class BatchGeneratorWithSceneMesh():
    def __init__(self, 
                dataset_path,
                device, 
                scene_verts_path,
                scene_sdf_path,
                mode='train', ## 'train' | 'test' for selecting different scene_sub_list
                read_all_to_ram=False):

        self.rec_list = list()
        self.index_rec = 0
        self.device = device

        if type(dataset_path) == str:

            self.dataset = h5py.File(dataset_path,'r')
            self.n_samples = self.dataset['depth'].shape[0]-1

            if not read_all_to_ram:
                self.depth_stream = self.dataset['depth']
                self.seg_stream = self.dataset['seg']
                self.body_stream = self.dataset['body']
                self.cam_ext_stream = self.dataset['cam_ext']
                self.cam_int_stream = self.dataset['cam_int']
                self.max_d_stream = self.dataset['max_d']
                self.sceneid_stream = self.dataset['sceneid']
            else:
                print('[INFO][BatchGeneratorWithSceneMesh] read all data to RAM..')
                self.depth_stream = self.dataset['depth'][1:]
                self.seg_stream = self.dataset['seg'][1:]
                self.body_stream = self.dataset['body'][1:]
                self.cam_ext_stream = self.dataset['cam_ext'][1:]
                self.cam_int_stream = self.dataset['cam_int'][1:]
                self.max_d_stream = self.dataset['max_d'][1:]
                self.sceneid_stream = self.dataset['sceneid'][1:]

        elif type(dataset_path) == list:
            list_depth_stream   = []
            list_seg_stream     = []
            list_body_stream    = []
            list_cam_ext_stream = []
            list_cam_int_stream = []
            list_max_d_stream   = []
            list_sceneid_stream = []
            print('[INFO][BatchGeneratorWithSceneMesh] realcams and virtualcams, read all data to RAM..')
            for dd in dataset_path:
                dataset = h5py.File(dd,'r')
                _ns = dataset['depth'].shape[0]
                for ii in range(1, _ns):
                    list_depth_stream  .append( dataset['depth'][ii:ii+1])
                    list_seg_stream    .append( dataset['seg'][ii:ii+1])
                    list_body_stream   .append( dataset['body'][ii:ii+1])
                    list_cam_ext_stream.append( dataset['cam_ext'][ii:ii+1])
                    list_cam_int_stream.append( dataset['cam_int'][ii:ii+1])
                    list_max_d_stream  .append( dataset['max_d'][ii:ii+1])
                    list_sceneid_stream.append( dataset['sceneid'][ii:ii+1])

            self.depth_stream =   np.concatenate(list_depth_stream  , axis=0)
            self.seg_stream =     np.concatenate(list_seg_stream    , axis=0)
            self.body_stream =    np.concatenate(list_body_stream   , axis=0)
            self.cam_ext_stream = np.concatenate(list_cam_ext_stream, axis=0)
            self.cam_int_stream = np.concatenate(list_cam_int_stream, axis=0)
            self.max_d_stream =   np.concatenate(list_max_d_stream  , axis=0)
            self.sceneid_stream = np.concatenate(list_sceneid_stream, axis=0)                

            self.n_samples = self.depth_stream.shape[0]


        
        self.scene_name_list = ['BasementSittingBooth','MPH1Library', 'MPH8', 'MPH11', 'MPH16',
        'MPH112', 'N0SittingBooth', 'N0Sofa', 'N3Library', 'N3Office',
        'N3OpenArea', 'Werkraum']

        if mode == 'train':
            scene_sub_list = ['BasementSittingBooth', 'MPH8', 'MPH11',
            'MPH112', 'N0Sofa', 'N3Library', 'N3Office', 'Werkraum']

        elif mode == 'test':
            scene_sub_list = ['MPH16', 'MPH1Library','N0SittingBooth', 'N3OpenArea']


        if mode != 'all':
            ## preserve data only in the sub scene idx
            print('[INFO][BatchGeneratorWithSceneMesh] select data in scene_sub_list..')
            scene_sub_id = [self.scene_name_list.index(x) for x in scene_sub_list]
            mask = [x in scene_sub_id for x in self.sceneid_stream]
            self.index = np.where(mask)[0].tolist()
            random.shuffle(self.index)
            if 0 in self.index:
                self.index.remove(0)
            
            self.n_samples = len(self.index)
            print('[INFO][BatchGeneratorWithSceneMesh] select data in scene_sub_list..Done')
            print('[INFO][BatchGeneratorWithSceneMesh] n_samples={:d}'.format(self.n_samples))

        else:
            self.index = list(range(self.n_samples))
            print('[INFO][BatchGeneratorWithSceneMesh] use all prox data...')
            print('[INFO][BatchGeneratorWithSceneMesh] n_samples={:d}'.format(self.n_samples))



        print('[INFO][BatchGeneratorWithSceneMesh] loading scene mesh files...')
        scene_file_list = glob.glob(os.path.join(scene_verts_path, '*'))
        self.scene_list = []
        self.min_n_faces = 1e8

        for scenefile in scene_file_list:
            scenename = scenefile.split("/")[-1][:-4]
            scene_o3d = o3d.io.read_triangle_mesh(scenefile)
            scene_verts = np.asarray(scene_o3d.vertices)
            scene_faces = np.asarray(scene_o3d.triangles)

            with open(os.path.join(scene_sdf_path, scenename+'.json')) as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'])
                grid_max = np.array(sdf_data['max'])
                grid_dim = sdf_data['dim']
            sdf = np.load(os.path.join(scene_sdf_path, scenename + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
            
            scene={}
            scene['name'] = scenename
            scene['verts'] = scene_verts
            scene['faces'] = scene_faces
            scene['grid_min']=grid_min
            scene['grid_max']=grid_max
            scene['grid_dim']=grid_dim
            scene['sdf'] = sdf

            if scene_faces.shape[0] < self.min_n_faces:
                self.min_n_faces = scene_faces.shape[0]

            self.scene_list.append(scene)
        print('[INFO][BatchGeneratorWithSceneMesh] loading scene mesh files...Done')



    def reset(self):
        self.index_rec = 0
        random.shuffle(self.index)
        self.n_samples = len(self.index)
        print('[INFO][BatchGeneratorWithSceneMesh] reset dataset')
        print('[INFO][BatchGeneratorWithSceneMesh] n_samples={:d}'.format(self.n_samples))


    def has_next_batch(self):
        if self.index_rec < self.n_samples:
            return True
        else:
            return False


    def next_batch(self, batch_size):

        s_verts_list = []
        s_faces_list = []
        s_grid_min_list = []
        s_grid_max_list = []
        s_grid_dim_list = []
        s_grid_sdf_list = []

        lb = self.index_rec
        ub = min(self.index_rec+batch_size, self.n_samples)
        self.index_rec = self.index_rec + batch_size
        
        if ub-lb < batch_size:
            return None

        idx_ = sorted(self.index[lb:ub])

        depth_batch = torch.tensor(self.depth_stream[idx_],dtype=torch.float32,device=self.device)
        seg_batch = torch.tensor(self.seg_stream[idx_],dtype=torch.float32,device=self.device)
        body_batch = torch.tensor(self.body_stream[idx_],dtype=torch.float32,device=self.device)
        cam_ext_batch = torch.tensor(self.cam_ext_stream[idx_],dtype=torch.float32,device=self.device)
        cam_int_batch = torch.tensor(self.cam_int_stream[idx_],dtype=torch.float32,device=self.device)
        max_d_batch = torch.tensor(self.max_d_stream[idx_],dtype=torch.float32,device=self.device)


        body_z_batch = body_batch[:,2]
        if torch.abs(body_z_batch).max() > torch.abs(max_d_batch).max():
            print('[INFO][BatchGeneratorWithSceneMesh] encounter wrong prox fitting')
            return None

        
        ## get scene info
        sceneid_batch = self.sceneid_stream[idx_].astype(np.int64)
        scenename_batch = [self.scene_name_list[ii] for ii in list(sceneid_batch)]


        for b_ in range(batch_size):
            scene= [x for x in self.scene_list if x['name']==scenename_batch[b_]][0]
            scene_verts = torch.tensor(scene['verts'], 
                                       dtype=torch.float32, 
                                       device=self.device).unsqueeze(0)
            scene_faces_idx = scene['faces'][:self.min_n_faces,:]
            scene_faces = torch.tensor(scene['verts'][scene_faces_idx],
                                       dtype=torch.float32, 
                                       device=self.device).unsqueeze(0)
            
            scene_grid_min = torch.tensor(scene['grid_min'], 
                                          dtype=torch.float32, 
                                          device=self.device).unsqueeze(0)
            scene_grid_max = torch.tensor(scene['grid_max'], 
                                          dtype=torch.float32, 
                                          device=self.device).unsqueeze(0)
            scene_grid_dim = torch.tensor(scene['grid_dim'], 
                                          dtype=torch.float32, 
                                          device=self.device).unsqueeze(0)
            scene_sdf = torch.tensor(scene['sdf'], 
                                     dtype=torch.float32, 
                                     device=self.device).unsqueeze(0)

            s_verts_list.append(scene_verts)
            s_faces_list.append(scene_faces)
            s_grid_min_list.append(scene_grid_min)
            s_grid_max_list.append(scene_grid_max)
            s_grid_dim_list.append(scene_grid_dim)
            s_grid_sdf_list.append(scene_sdf)

        s_verts_batch = torch.cat(s_verts_list,  dim=0)
        s_faces_batch = torch.cat(s_faces_list, dim=0)
        s_grid_min_batch = torch.cat(s_grid_min_list, dim=0)
        s_grid_max_batch = torch.cat(s_grid_max_list, dim=0)
        s_grid_dim_batch = torch.cat(s_grid_dim_list, dim=0)
        s_grid_sdf_batch = torch.cat(s_grid_sdf_list, dim=0)



        return [depth_batch, seg_batch, body_batch, cam_ext_batch, 
                cam_int_batch,max_d_batch,
                s_verts_batch, s_faces_batch,
                s_grid_min_batch, s_grid_max_batch, s_grid_dim_batch, 
                s_grid_sdf_batch]















def inverse_trans(trans):
    
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    
    
    return mat



class BatchGeneratorWithSceneMeshMatfile(object):
    '''
    this class is only used to convert a large number of matfiles into hdf5 files
    '''
    def __init__(self, 
                dataset_path,
                device, 
                scene_verts_path,
                scene_sdf_path):

        self.rec_list = list()
        self.index_rec = 0
        self.rec_list = glob.glob(os.path.join(dataset_path, '*/*.mat'))
        self.device = device
        random.shuffle(self.rec_list)
        self.n_samples = len(self.rec_list)

        ### read all the scene files to memory
        scene_file_list = glob.glob(os.path.join(scene_verts_path, '*'))
        self.scene_list = []
        self.min_n_faces = 1e8

        for scenefile in scene_file_list:
            scenename = scenefile.split("/")[-1][:-4]
            scene_o3d = o3d.io.read_triangle_mesh(scenefile)
            scene_verts = np.asarray(scene_o3d.vertices)
            scene_faces = np.asarray(scene_o3d.triangles)

            with open(os.path.join(scene_sdf_path, scenename+'.json')) as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'])
                grid_max = np.array(sdf_data['max'])
                grid_dim = sdf_data['dim']
            sdf = np.load(os.path.join(scene_sdf_path, scenename + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
            

            scene={}
            scene['name'] = scenename
            scene['verts'] = scene_verts
            scene['faces'] = scene_faces
            scene['grid_min']=grid_min
            scene['grid_max']=grid_max
            scene['grid_dim']=grid_dim
            scene['sdf'] = sdf

            if scene_faces.shape[0] < self.min_n_faces:
                self.min_n_faces = scene_faces.shape[0]

            self.scene_list.append(scene)


    def reset(self):
        self.index_rec = 0
        random.shuffle(self.rec_list)


    def has_next_batch(self):
        if self.index_rec < len(self.rec_list):
            return True
        else:
            return False



    def data_preprocessing(self, img, modality, target_domain_size=[128, 128], filename=None):

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
            # img[img==np.nan]=0.0

        if modality == 'seg':
            img[img>41] = 41



        ## rescale to [-1,1]
        # _img = (img-np.mean(img)) / np.std(img)
        max_val = torch.max(img)
        _img = 2* img / max_val - 1.0

        # print(_img)
        # if _img[0] > 1e3:
        #     print(filename)
        #     pdb.set_trace()

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
            
            # print(img_resize)

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



    def scipy_matfile_parse(self, filename):
        '''
        parse data from files and put them to GPU
        '''
        data = sio.loadmat(filename)

        ## get depth and seg. Perform preprocessing
        if 'virtualcam' in filename:
            depth = torch.tensor(data['depth'], dtype=torch.float32, device=self.device).view(1,1,128,128)
            seg = torch.tensor(data['seg'], dtype=torch.float32, device=self.device).view(1,1,128,128)
            depth0 = torch.tensor(data['depth0'], dtype=torch.float32, device=self.device)
            max_depth = torch.max(depth0)

        else:
            depth0_np = data['depth']
            seg_np = data['seg']

            depth0 = torch.tensor(depth0_np, dtype=torch.float32, device=self.device)
            seg = torch.tensor(seg_np, dtype=torch.float32, device=self.device)

            depth, s_, max_depth = self.data_preprocessing(depth0, 'depth', target_domain_size=[128, 128],
                                                        filename=filename)
            seg, _, _ = self.data_preprocessing(seg, 'seg', target_domain_size=[128, 128],
                                                        filename=filename)


        ## get cam
        cam_intrinsic_np = data['cam'][0][0]['intrinsic']
        cam_intrinsic = torch.tensor(cam_intrinsic_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        cam_extrinsic_np = data['cam'][0][0]['extrinsic']
        cam_extrinsic_np = inverse_trans(cam_extrinsic_np)
        cam_extrinsic = torch.tensor(cam_extrinsic_np, dtype=torch.float32, device=self.device).unsqueeze(0)


        ## get body params and pre-processing the global translation
        body_t = data['body'][0][0]['transl']
        body_r = data['body'][0][0]['global_orient']
        body_shape=data['body'][0][0]['betas']
        body_pose = data['body'][0][0]['pose_embedding']
        body_lhp = data['body'][0][0]['left_hand_pose']
        body_rhp = data['body'][0][0]['right_hand_pose']
        body_np = np.concatenate([body_t, body_r, body_shape, 
                               body_pose, body_lhp, body_rhp],axis=-1)
        body = torch.tensor(body_np, dtype=torch.float32, device=self.device)


        scene = [x for x in self.scene_list if x['name'] in filename][0]
        scene_verts = torch.tensor(scene['verts'], dtype=torch.float32, device=self.device).unsqueeze(0)




        scene_faces_idx = scene['faces'][:self.min_n_faces,:]
        scene_faces = torch.tensor(scene['verts'][scene_faces_idx],
                                   dtype=torch.float32, 
                                   device=self.device).unsqueeze(0)
        
        scene_grid_min = torch.tensor(scene['grid_min'], dtype=torch.float32, device=self.device).unsqueeze(0)
        scene_grid_max = torch.tensor(scene['grid_max'], dtype=torch.float32, device=self.device).unsqueeze(0)
        scene_grid_dim = torch.tensor(scene['grid_dim'], dtype=torch.float32, device=self.device).unsqueeze(0)
        scene_sdf = torch.tensor(scene['sdf'], dtype=torch.float32, device=self.device).unsqueeze(0)

        return [depth, seg, body, cam_extrinsic, cam_intrinsic, max_depth.view(1),
                scene_verts, scene_faces, 
                scene_grid_min, scene_grid_max, scene_grid_dim, scene_sdf]


    def next_batch(self, batch_size):
        d_list = []
        s_list = []
        b_list = []
        cam_ext_list = []
        cam_int_list = []
        max_depth_list = []
        s_verts_list = []
        s_faces_list = []
        s_grid_min_list = []
        s_grid_max_list = []
        s_grid_dim_list = []
        s_grid_sdf_list = []
        filename_list = []

        b_ = 0
        while b_ < batch_size:

        
            filename = self.rec_list[self.index_rec]
            self.index_rec += 1

            if not self.has_next_batch():
                return None
        
            [depth, seg, body, cam_extrinsic, cam_intrinsic,max_d,
             scene_verts, scene_faces, 
             scene_grid_min, scene_grid_max, 
             scene_grid_dim, scene_sdf] = self.scipy_matfile_parse(filename)
            
            if torch.sum(torch.isnan(body)) > 0:
                continue

            if torch.abs(body[0][0]) > 10:
                continue

            filename_list.append(filename)
            d_list.append(depth)
            s_list.append(seg)
            b_list.append(body)
            cam_ext_list.append(cam_extrinsic)
            cam_int_list.append(cam_intrinsic)
            max_depth_list.append(max_d)
            s_verts_list.append(scene_verts)
            s_faces_list.append(scene_faces)
            s_grid_min_list.append(scene_grid_min)
            s_grid_max_list.append(scene_grid_max)
            s_grid_dim_list.append(scene_grid_dim)
            s_grid_sdf_list.append(scene_sdf)
            b_ += 1


        depth_batch =  torch.cat(d_list,   dim=0)
        seg_batch =    torch.cat(s_list,   dim=0)
        body_batch =   torch.cat(b_list,   dim=0)
        cam_ext_batch = torch.cat(cam_ext_list,   dim=0)
        cam_int_batch = torch.cat(cam_int_list,   dim=0)
        max_d_batch = torch.cat(max_depth_list,   dim=0)
        s_verts_batch = torch.cat(s_verts_list,  dim=0)
        s_faces_batch =    torch.cat(s_faces_list, dim=0)
        s_grid_min_batch =    torch.cat(s_grid_min_list, dim=0)
        s_grid_max_batch =    torch.cat(s_grid_max_list, dim=0)
        s_grid_dim_batch =    torch.cat(s_grid_dim_list, dim=0)
        s_grid_sdf_batch =    torch.cat(s_grid_sdf_list, dim=0)



        if torch.sum(torch.isnan(depth_batch)) > 0:
            print('[ERROR] nan in depth_batch')
            return None

        if torch.sum(torch.isnan(seg_batch)) > 0:
            print('[ERROR] nan in seg_batch')
            return None
        

        if torch.sum(torch.isnan(body_batch)) > 0:
            print('[ERROR] nan in body_batch')
            return None



        return [depth_batch, seg_batch, body_batch, cam_ext_batch, cam_int_batch,max_d_batch,
                s_verts_batch, s_faces_batch,
                s_grid_min_batch, s_grid_max_batch, s_grid_dim_batch, s_grid_sdf_batch, filename_list]






















class BatchGeneratorTest(object):
    def __init__(self, dataset_path, device):
        ## note that dataset_path specifies the scene

        self.rec_list = list()
        self.index_rec = 0
        self.rec_list = glob.glob(os.path.join(dataset_path, '*.mat'))
        self.device = device
        random.shuffle(self.rec_list)

    def reset(self):
        self.index_rec = 0
        random.shuffle(self.rec_list)

    def has_next_batch(self):
        if self.index_rec < len(self.rec_list):
            return True
        else:
            return False


    def data_preprocessing(self, img, modality, target_domain_size=[128, 128], filename=None):

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


        ## rescale to [-1,1]
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



    def scipy_matfile_parse(self, filename):
        '''
        parse data from files and put them to GPU
        '''
        data = sio.loadmat(filename)
        depth0_np = data['depth']
        seg_np = data['seg']

        ## change them to torch tensor
        depth0 = torch.tensor(depth0_np, dtype=torch.float32, device=self.device)
        seg = torch.tensor(seg_np, dtype=torch.float32, device=self.device)

        ## pre_processing
        depth, _, max_d = self.data_preprocessing(depth0, 'depth', target_domain_size=[128, 128])
        seg, _ ,_ = self.data_preprocessing(seg, 'seg', target_domain_size=[128, 128])


        ## get camera parameters
        cam_intrinsic_np = data['cam'][0][0]['intrinsic']
        cam_intrinsic = torch.tensor(cam_intrinsic_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        cam_extrinsic_np = data['cam'][0][0]['extrinsic']
        cam_extrinsic_np = np.linalg.inv(cam_extrinsic_np)
        cam_extrinsic = torch.tensor(cam_extrinsic_np, dtype=torch.float32, device=self.device).unsqueeze(0)



        ## get body params and pre-processing the global translation
        body_t = data['body'][0][0]['transl']
        body_r = data['body'][0][0]['global_orient']
        body_shape=data['body'][0][0]['betas']
        body_pose = data['body'][0][0]['body_pose']
        body_lhp = data['body'][0][0]['left_hand_pose']
        body_rhp = data['body'][0][0]['right_hand_pose']
        body_np = np.concatenate([body_t, body_r, body_shape, 
                               body_pose, body_lhp, body_rhp],axis=-1)
        body = torch.tensor(body_np, dtype=torch.float32, device=self.device)


        return depth, seg, max_d.view(1), cam_intrinsic, cam_extrinsic, body

        
    def next_batch(self, batch_size):
        d_list = []
        s_list = []
        max_d_list = []
        cam_int_list = []
        cam_ext_list = []
        body_list = []

        for _ in range(batch_size):
            filename = self.rec_list[0]

            if not self.has_next_batch():
                return None
            depth, seg, max_d, cam_intrinsic, cam_extrinsic,body = self.scipy_matfile_parse(filename)
            
            d_list.append(depth)
            s_list.append(seg)
            max_d_list.append(max_d)
            cam_int_list.append(cam_intrinsic)
            cam_ext_list.append(cam_extrinsic)
            body_list.append(body)

        depth_batch =  torch.cat(d_list,   dim=0)
        seg_batch =    torch.cat(s_list,   dim=0)
        max_d_batch = torch.cat(max_d_list,  dim=0)
        cam_int_batch =    torch.cat(cam_int_list, dim=0)
        cam_ext_batch =    torch.cat(cam_ext_list, dim=0)
        body_batch =    torch.cat(body_list, dim=0)


        if torch.sum(torch.isnan(depth_batch)) > 0:
            print('[ERROR] nan in depth_batch')
            return None

        if torch.sum(torch.isnan(seg_batch)) > 0:
            print('[ERROR] nan in seg_batch')
            return None
               

        return depth_batch, seg_batch, max_d_batch, cam_int_batch, cam_ext_batch, body_batch




################################### unit test #################################
if __name__=='__main__':
    # batch_gen = BatchGeneratorForBaseline(dataset_path='/is/cluster/work/yzhang/PROX/realcams.hdf5',
    #                                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #                                   mode='train',
    #                                   read_all_to_ram=False)


    batch_gen = BatchGeneratorWithSceneMesh(dataset_path='/is/cluster/yzhang/PROX/virtualcams3.hdf5',
                                            scene_verts_path = '/is/cluster/work/yzhang/PROX/scenes_downsampled',
                                            scene_sdf_path = '/is/cluster/work/yzhang/PROX/scenes_sdf',
                                            mode='train',
                                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                            read_all_to_ram=False)

    while batch_gen.has_next_batch():
        data = batch_gen.next_batch(4)

        pdb.set_trace()
        # if data == None:
        #     pdb.set_trace()
        # else:
        #     