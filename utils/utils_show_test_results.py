'''
- this script is to show the generated results in PROXE
- the users need first to run test*.py
- one can uncomment the code for interactive visualization in open3d
'''

import os, glob
import os.path as osp
import cv2
import numpy as np
import json
import yaml
import open3d as o3d
import trimesh
import argparse
import matplotlib.pyplot as plt

import torch
import pickle
import smplx

import sys
sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal')

from human_body_prior.tools.model_loader import load_vposer
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.io as sio




def hex2rgb(hex_color_list):
    rgb_list = []
    for hex_color in hex_color_list:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i+2], 16) for i in (0, 2, 4))
        rgb_list.append(rgb)

    return np.array(rgb_list)






def update_cam(cam_param,  trans):
    
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat
    
    return cam_param


def capture_image(vis, outfilename=None):
    image = vis.capture_screen_float_buffer()
    if outfilename is None:
        plt.imshow(np.asarray(image))
        plt.show()
    else:
        plt.imsave(outfilename, np.asarray(image))
        print('-- output image to:' + outfilename)
    return False


def get_trans_mat(R, T):
    mat_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([R, T.reshape([3,1])],axis=-1)
    mat = np.concatenate([mat, mat_aux],axis=0)

    return mat



def main(args):

    scene_name = os.path.abspath(args.gen_folder).split("/")[-1]

    data_dir = args.prox_dir
    cam2world_dir = osp.join(data_dir, 'cam2world')
    scene_dir = osp.join(data_dir, 'scenes_semantics')
    

    outimg_dir = args.outimg_dir
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)


    ### setup visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face=True
    

    ### put the scene into the environment
    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    

    trans = np.eye(4)
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))
    vis.add_geometry(scene)
    trans2 = np.eye(4)
    trans2[:3,:3] = np.array([[1,0,0], [0, -1,0], [0,0,-1]])
    trans2[:3, -1] = np.array([0,0,3.5])
    if scene_name == 'N0SittingBooth':
        trans2[:3, -1] += np.array([2,0,0])


    # put the body into the environment
    vposer_ckpt = osp.join(args.model_folder, 'vposer_v1_0')
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')

    model = smplx.create(args.model_folder, model_type='smplx',
                         gender=args.gender, ext='npz',
                         num_pca_comps=args.num_pca_comps,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True
                         )


    ## create a corn at the camera location
    # mesh_corn = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # mesh_corn.transform(trans)
    # vis.add_geometry(mesh_corn)
    # vis.update_geometry()
    # print(trans)

    ## create a corn at the world origin
    mesh_corn2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh_corn2)
    vis.update_geometry(mesh_corn2)
    # print(trans)

    cv2.namedWindow("GUI")

    gen_file_list = sorted(glob.glob(os.path.join(args.gen_folder, '*')))

    body = o3d.geometry.TriangleMesh()
    vis.add_geometry(body)
    for idx, gen_file in enumerate(gen_file_list):

        with open(gen_file, 'rb') as f:
            param = pickle.load(f)
        
        cam_ext = param['cam_ext'][0]
        cam_int = param['cam_int'][0]
        body_pose = vposer.decode(torch.tensor(param['body_pose']), output_type='aa').view(1, -1)
        torch_param = {}
        
        for key in param.keys():
            if key in ['body_pose', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])

        
        output = model(return_verts=True, body_pose=body_pose, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()


        body.vertices = o3d.utility.Vector3dVector(vertices)
        body.triangles = o3d.utility.Vector3iVector(model.faces)
        body.vertex_normals = o3d.utility.Vector3dVector([])
        body.triangle_normals = o3d.utility.Vector3dVector([])
        body.compute_vertex_normals()
        body.transform(cam_ext) # or directly using trans

        vis.update_geometry(body)

        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param = update_cam(cam_param, trans)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        # vis.run()

        capture_image(vis, outfilename=os.path.join(outimg_dir, 'img_{:06d}_cam1.png'.format(idx)))

        ### setup rendering cam, depth capture, segmentation capture    
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param = update_cam(cam_param, trans2)
        ctr.convert_from_pinhole_camera_parameters(cam_param)


        vis.poll_events()
        vis.update_renderer()

        capture_image(vis, outfilename=os.path.join(outimg_dir, 'img_{:06d}_cam2.png'.format(idx)))
#        if idx >= 399:
#            break
        # vis.destroy_window()
        # while True:
        #     vis.poll_events()
        #     vis.update_renderer()
        #     cv2.imshow("GUI", np.random.random([10,10,3]))
        #     key = cv2.waitKey(15)
        #     if key == 27:
        #         break

    # ## read frame image
    # color_img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))
    # color_img = color_img[:,::-1,:]



    # while True:
    #     cv2.imshow('frame', color_img)
    #     cv2.imshow('scene_depth',
    #                 cv2.normalize(depth, 
    #                     None, 0,255, 
    #                     norm_type = cv2.NORM_MINMAX, 
    #                     dtype = cv2.CV_32F).astype(np.uint8)
    #                 )
    #     cv2.imshow('scene_seg', seg)
    # vis.poll_events()
    # vis.update_renderer()

    #     vis2.poll_events()
    #     vis2.update_renderer()
    
    #     key = cv2.waitKey(15)
    #     if key == 27:
    #         break

if __name__ == '__main__':

    import sys
    scenes = ['MPH16', 'N0SittingBooth', 'MPH1Library', 'N3OpenArea']

    test_folder = sys.argv[1]

    for ss in scenes:

        gen_folder = os.path.join(test_folder, 'virtualrealcams', ss)
        outimg_dir = os.path.join(test_folder, 'virtualrealcams_img', ss)

        args = type('', (), {})()
        args.gen_folder = gen_folder
        args.outimg_dir = outimg_dir
        args.prox_dir = '/home/yzhang/Videos/PROXE'
        args.model_folder = '/home/yzhang/body_models/VPoser/'
        args.num_pca_comps = 12
        args.gender = 'neutral'


        main(args)
