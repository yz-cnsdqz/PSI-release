import os, sys
import os.path as osp
import cv2
import numpy as np
import json
import yaml
import open3d as o3d
import trimesh
import argparse
import matplotlib.pyplot as plt

sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal')

import torch
import pickle
import smplx
from human_body_prior.tools.model_loader import load_vposer
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.io as sio
import glob



def hex2rgb(hex_color_list):
    rgb_list = []
    for hex_color in hex_color_list:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i+2], 16) for i in (0, 2, 4))
        rgb_list.append(rgb)

    return np.array(rgb_list)




def color_encoding(mesh):
    ''' 
    we use the color coding of Matterport3D
    '''
    
    ## get the color coding from Matterport3D
    matter_port_label_filename = '/is/ps2/yzhang/Pictures/Matterport/metadata/mpcat40.tsv'
    df = pd.DataFrame()
    df = pd.read_csv(matter_port_label_filename,sep='\t')
    color_coding_hex = list(df['hex']) # list of str
    color_coding_rgb = hex2rgb(color_coding_hex)

    ## update the mesh vertex color accordingly
    verid = np.mean(np.asarray(mesh.vertex_colors)*255/5.0,axis=1).astype(int)
    verid[verid>=41]=41
    vercolor = np.take(color_coding_rgb, list(verid), axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vercolor/255.0)

    return mesh




def update_cam(cam_param,  trans):
    
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat
    
    return cam_param




def get_trans_mat(R, T):
    mat_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([R, T.reshape([3,1])],axis=-1)
    mat = np.concatenate([mat, mat_aux],axis=0)

    return mat




def main(args):
    fitting_dir = args.fitting_dir
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = osp.join(fitting_dir, 'results')
    data_dir = args.data_dir
    cam2world_dir = osp.join(data_dir, 'cam2world')
    scene_dir = osp.join(data_dir, 'scenes_semantics')
    recording_dir = osp.join(data_dir, 'recordings', recording_name)
    color_dir = os.path.join(recording_dir, 'Color')
    scene_name = os.path.abspath(recording_dir).split("/")[-1].split("_")[0]

    output_folder = os.path.join('/mnt/hdd','PROX','realcams_v3')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    ### setup visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=270,visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face=True


    ### put the scene into the environment
    if scene_name in ['MPH112', 'MPH16']:
        scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '_withlabels_OpenAWall.ply'))
    else:
        scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '_withlabels.ply'))


    trans = np.eye(4)
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))
    vis.add_geometry(scene)




    ### setup rendering cam, depth capture, segmentation capture    
    ctr = vis.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    cam_param = update_cam(cam_param, trans)
    ctr.convert_from_pinhole_camera_parameters(cam_param)

    ## capture depth image
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    _h = depth.shape[0]
    _w = depth.shape[1]
    factor = 4
    depth = cv2.resize(depth, (_w//factor, _h//factor))

    ## capture semantics
    seg = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    verid = np.mean(seg*255/5.0,axis=-1) #.astype(int)
    verid = cv2.resize(verid, (_w//factor, _h//factor))



    ## get render cam parameters
    cam_dict = {}
    cam_dict['extrinsic'] = cam_param.extrinsic
    cam_dict['intrinsic'] = cam_param.intrinsic.intrinsic_matrix
    
    count = 0
    for img_name in sorted(os.listdir(fitting_dir))[::15]:
        print('viz frame {}'.format(img_name))

        ## get humam body params
        filename =osp.join(fitting_dir, img_name, '000.pkl')
        if not os.path.exists(filename):
            continue


        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)


        body_dict={}
        for key, val in param.items():
            if key in ['camera_rotation', 'camera_translation', 
                       'jaw_pose', 'leye_pose','reye_pose','expression']:
                continue

            else:
                body_dict[key]=param[key]


        ## save depth, semantics and render cam
        outname1 = os.path.join(output_folder,recording_name) 
        if not os.path.exists(outname1):
            os.mkdir(outname1)
        
        outname = os.path.join(outname1, 'rec_{:06d}.mat'.format(count))
        
        ot_dict={}
        ot_dict['scaling_factor']=factor
        ot_dict['depth']=depth
        ot_dict['seg'] = verid
        ot_dict['cam'] = cam_dict
        ot_dict['body'] = body_dict
        sio.savemat(outname, ot_dict)
        
        count += 1

        


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':

    fitting_dir_list = glob.glob('/mnt/hdd/PROX/PROXD/*')
    data_dir = '/mnt/hdd/PROX'
    model_folder  = '/home/yzhang/body_models/VPoser/'
    
    args= {}

    for fitting_dir in fitting_dir_list:
        print('-- process {}'.format(fitting_dir))
        args['fitting_dir'] = fitting_dir
        args['data_dir'] = data_dir
        args['model_folder'] = model_folder

        main(Struct(**args))
