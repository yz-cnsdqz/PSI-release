import numpy as np
import scipy.io as sio
import os, glob, sys
import h5py_cache as h5c

sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal')
sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal/source')


from batch_gen_hdf5 import BatchGeneratorWithSceneMeshMatfile

import torch


'''
In this script, we put all mat files into a hdf5 file, so as to speed up the data loading process.
'''


dataset_path = '/mnt/hdd/PROX/snapshot_realcams_v3'
outfilename = 'realcams.hdf5'

h5file_path = os.path.join('/home/yzhang/Videos/PROXE', outfilename)


batch_gen = BatchGeneratorWithSceneMeshMatfile(dataset_path=dataset_path,
                                                scene_verts_path = '/home/yzhang/Videos/PROXE/scenes_downsampled',
                                                scene_sdf_path = '/home/yzhang/Videos/PROXE/scenes_sdf',
                                                device=torch.device('cuda'))


### create the dataset used in the hdf5 file
with h5c.File(h5file_path, mode='w',chunk_cache_mem_size=1024**2*128) as hdf5_file:
    while batch_gen.has_next_batch():
        train_data = batch_gen.next_batch(1)
                    
        if train_data is None:
            continue


        train_data_np = [x.detach().cpu().numpy() for x in train_data[:-1]]
        break




    [depth_batch, seg_batch, body_batch, 
        cam_ext_batch, cam_int_batch, max_d_batch,
        s_verts_batch, s_faces_batch,
        s_grid_min_batch, s_grid_max_batch, 
        s_grid_dim_batch, s_grid_sdf_batch] = train_data_np

    n_samples = batch_gen.n_samples
    print('-- n_samples={:d}'.format(n_samples))
    hdf5_file.create_dataset("sceneid", shape=(1,), chunks=True, dtype=np.float32, maxshape=(None,) )
    hdf5_file.create_dataset("depth",       shape=(1,)+tuple(depth_batch.shape[1:])      ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(depth_batch.shape[1:])     )
    hdf5_file.create_dataset("seg",         shape=(1,)+tuple(seg_batch.shape[1:])        ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(seg_batch.shape[1:])       )
    hdf5_file.create_dataset("body",        shape=(1,)+tuple(body_batch.shape[1:])       ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(body_batch.shape[1:])      )
    hdf5_file.create_dataset("cam_ext",     shape=(1,)+tuple(cam_ext_batch.shape[1:])    ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(cam_ext_batch.shape[1:])   )
    hdf5_file.create_dataset("cam_int",     shape=(1,)+tuple(cam_int_batch.shape[1:])    ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(cam_int_batch.shape[1:])   )
    hdf5_file.create_dataset("max_d",       shape=(1,)+tuple(max_d_batch.shape[1:])      ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(max_d_batch.shape[1:])     )
    # hdf5_file.create_dataset("s_verts",     shape=(1,)+tuple(s_verts_batch.shape[1:])    ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_verts_batch.shape[1:])   )
    # hdf5_file.create_dataset("s_faces",     shape=(1,)+tuple(s_faces_batch.shape[1:])    ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_faces_batch.shape[1:])   )
    # hdf5_file.create_dataset("s_grid_min",  shape=(1,)+tuple(s_grid_min_batch.shape[1:]) ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_grid_min_batch.shape[1:]))
    # hdf5_file.create_dataset("s_grid_max",  shape=(1,)+tuple(s_grid_max_batch.shape[1:]) ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_grid_max_batch.shape[1:]))
    # hdf5_file.create_dataset("s_grid_dim",  shape=(1,)+tuple(s_grid_dim_batch.shape[1:]) ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_grid_dim_batch.shape[1:]))
    # hdf5_file.create_dataset("s_grid_sdf",  shape=(1,)+tuple(s_grid_sdf_batch.shape[1:]) ,chunks = True, dtype=np.float32, maxshape=(None,)+tuple(s_grid_sdf_batch.shape[1:]))


    batch_gen.reset()
    scene_list = ['BasementSittingBooth','MPH1Library', 'MPH8', 'MPH11', 'MPH16',
                    'MPH112', 'N0SittingBooth', 'N0Sofa', 'N3Library', 'N3Office',
                    'N3OpenArea', 'Werkraum'] # !!!! important!cat 

    ### create the dataset used in the hdf5 file
    idx = -1
    while batch_gen.has_next_batch():
        train_data = batch_gen.next_batch(1)
                    
        if train_data is None:
            continue

        [depth_batch, seg_batch, body_batch, 
            cam_ext_batch, cam_int_batch, max_d_batch,
            s_verts_batch, s_faces_batch,
            s_grid_min_batch, s_grid_max_batch, 
            s_grid_dim_batch, s_grid_sdf_batch, filename_list] = train_data

        ## check unavaliable prox fitting
        body_z_batch = body_batch[:,2]
        
        if body_z_batch.abs().max() >= max_d_batch.abs().max():
            print('-- encountered bad prox fitting. Skip it')
            continue
        

        if body_z_batch.min() <=0:
            print('-- encountered bad prox fitting. Skip it')
            continue


        idx = idx+1

        print('-- processing batch idx {:d}'.format(idx))

        filename = filename_list[0]
        scenename = filename.split('/')[-2].split('_')[0]
        sid = [scene_list.index(scenename)]


        hdf5_file["sceneid"].resize((hdf5_file["sceneid"].shape[0]+1, ))
        hdf5_file["sceneid"][-1,...] = sid[0]

        hdf5_file["depth"].resize((hdf5_file["depth"].shape[0]+1, )+hdf5_file["depth"].shape[1:])
        hdf5_file["depth"][-1,...] = depth_batch[0].detach().cpu().numpy()

        hdf5_file["seg"].resize((hdf5_file["seg"].shape[0]+1, )+hdf5_file["seg"].shape[1:])
        hdf5_file["seg"][-1,...] = seg_batch[0].detach().cpu().numpy()
        
        hdf5_file["body"].resize((hdf5_file["body"].shape[0]+1, )+hdf5_file["body"].shape[1:])
        hdf5_file["body"][-1,...] = body_batch[0].detach().cpu().numpy()
        
        hdf5_file["cam_ext"].resize((hdf5_file["cam_ext"].shape[0]+1, )+hdf5_file["cam_ext"].shape[1:])
        hdf5_file["cam_ext"][-1,...] = cam_ext_batch[0].detach().cpu().numpy()
        
        hdf5_file["cam_int"].resize((hdf5_file["cam_int"].shape[0]+1, )+hdf5_file["cam_int"].shape[1:])
        hdf5_file["cam_int"][-1,...] = cam_int_batch[0].detach().cpu().numpy()
        
        hdf5_file["max_d"].resize((hdf5_file["max_d"].shape[0]+1, )+hdf5_file["max_d"].shape[1:])
        hdf5_file["max_d"][-1,...] = max_d_batch[0].detach().cpu().numpy()

        # hdf5_file["s_verts"].resize((hdf5_file["s_verts"].shape[0]+1, )+hdf5_file["s_verts"].shape[1:])
        # hdf5_file["s_verts"][-1,...] = s_verts_batch[0].detach().cpu().numpy()
        
        # hdf5_file["s_faces"].resize((hdf5_file["s_faces"].shape[0]+1, )+hdf5_file["s_faces"].shape[1:])
        # hdf5_file["s_faces"][-1,...] = s_faces_batch[0].detach().cpu().numpy()

        # hdf5_file["s_grid_min"].resize((hdf5_file["s_grid_min"].shape[0]+1, )+hdf5_file["s_grid_min"].shape[1:])
        # hdf5_file["s_grid_min"][-1,...] = s_grid_min_batch[0].detach().cpu().numpy()
        
        # hdf5_file["s_grid_max"].resize((hdf5_file["s_grid_max"].shape[0]+1, )+hdf5_file["s_grid_max"].shape[1:])
        # hdf5_file["s_grid_max"][-1,...] = s_grid_max_batch[0].detach().cpu().numpy()

        # hdf5_file["s_grid_dim"].resize((hdf5_file["s_grid_dim"].shape[0]+1, )+hdf5_file["s_grid_dim"].shape[1:])
        # hdf5_file["s_grid_dim"][-1,...] = s_grid_dim_batch[0].detach().cpu().numpy()
        
        # hdf5_file["s_grid_sdf"].resize((hdf5_file["s_grid_sdf"].shape[0]+1, )+hdf5_file["s_grid_sdf"].shape[1:])
        # hdf5_file["s_grid_sdf"][-1,...] = s_grid_sdf_batch[0].detach().cpu().numpy()

    print('--file converting finish')



