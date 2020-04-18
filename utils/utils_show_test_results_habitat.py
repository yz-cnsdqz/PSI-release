'''
- this script is to show the generated results in Habitat
- the users need first to run test_habitat*.py
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
from human_body_prior.tools.model_loader import load_vposer
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.io as sio









#### the following camera extrinsics are for visualization purposes.
trans2_dict = {}
trans2_dict['17DRP5sb8fy-bedroom'] = np.array([[ 0.99769666, -0.03318259,  0.05916328,  9.39492349],
                             [-0.0349965,  -0.99894032,  0.02989132,  0.8169383 ],
                             [ 0.05810872, -0.03189298, -0.99780069,  1.76649009],
                             [ 0.   ,       0.   ,       0.    ,      1.        ]])


trans2_dict['17DRP5sb8fy-familyroomlounge'] = np.array([[-0.93535523, -0.0177688 , -0.35326315 ,-8.55068677],
                                                     [ 0.09580802, -0.97412908, -0.20467866 , 1.79405807],
                                                     [-0.34048702, -0.2252927 ,  0.91285913 ,-1.31050597],
                                                     [ 0.  ,        0.   ,       0.   ,       1.        ]]
                                                        )




trans2_dict['17DRP5sb8fy-livingroom'] = np.array([[ 0.7063483,  -0.11477746, -0.69849711,  1.33807416],
                                                 [ 0.09810802, -0.96136956 , 0.25718358 , 1.87543173],
                                                 [-0.70103274, -0.25018935, -0.66780116, -1.70209309],
                                                 [ 0.    ,      0.     ,     0.   ,       1.        ]])


trans2_dict['sKLMLpTHeUy-familyname_0_1'] = np.array([[ 0.05682247, -0.01931449 , 0.99819745 , 0.06459968],
                                                     [-0.07957467, -0.99671968, -0.0147561,  -1.48962379],
                                                     [ 0.99520806, -0.07859275, -0.05817301  ,1.8495453 ],
                                                     [ 0.    ,      0.   ,       0.    ,      1.        ]]
                                                        )



trans2_dict['X7HyMhZNoso-livingroom_0_16'] = np.array([[-0.68180289, -0.05110302, -0.72974879,  7.1079669 ],
                                                         [ 0.56427749, -0.67158339, -0.48017357,  8.29411821],
                                                         [-0.46554885, -0.73916455,  0.48672379, -4.96794284],
                                                         [ 0.    ,      0.    ,      0.  ,        1.        ]]
                                                            )



trans2_dict['zsNo4HB9uLZ-bedroom0_0'] = np.array([[ 0.72913437, -0.05539176, -0.68212523, -4.10728367],
                                                 [ 0.44736699, -0.71570109,  0.53631588,  4.02113353],
                                                 [-0.51790525, -0.69620665 ,-0.49706182, -0.06188668],
                                                 [ 0.   ,       0.  ,        0.      ,    1.        ]]
                                                    )



trans2_dict['zsNo4HB9uLZ-livingroom0_13'] = np.array([[-9.95373824e-01, -4.65599127e-02,  8.40423952e-02,  6.11471871e+00],
                                                     [ 4.67419759e-02, -9.98906977e-01,  1.98919308e-04 , 8.17973221e-01],
                                                     [ 8.39412732e-02,  4.12630668e-03,  9.96462160e-01 , 8.93803983e-01],
                                                     [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]]
                                                        )




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


def get_pcd_from_depth(depth0, cam):
    depth0 = cv2.resize(depth0, tuple(  (cam[:2, -1]*2).astype(np.int)   ))
    depthp = o3d.geometry.Image(depth0)

    camp = o3d.camera.PinholeCameraIntrinsic()
    camp.intrinsic_matrix=cam

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth0, camp, depth_scale=1)

    return pcd


def main(args):

    scene_name = os.path.abspath(args.gen_folder).split("/")[-1]
    
    outimg_dir = args.outimg_dir
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)


    ### setup visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face=True
    

    ### put the scene into the environment
    scene = o3d.io.read_triangle_mesh(osp.join(args.prox_dir, scene_name + '.ply'))
    vis.add_geometry(scene)
    vis.update_geometry()



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
    # mesh_corn2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # vis.add_geometry(mesh_corn2)
    # vis.update_geometry()
    # print(trans)



    cv2.namedWindow("GUI")

    gen_file_list = glob.glob(os.path.join(args.gen_folder, '*'))

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
        T_mat = np.eye(4)
        T_mat[1,:] = np.array([0,-1,0,0])
        T_mat[2,:] = np.array([0,0,-1,0])
        trans = np.dot(cam_ext,T_mat)
        body.transform(trans)
        vis.update_geometry()



        # while True:
        #     vis.poll_events()
        #     vis.update_renderer()
        #     cv2.imshow("GUI", np.random.random([10,10,3]))

        #     # ctr = vis.get_view_control()
        #     # cam_param = ctr.convert_to_pinhole_camera_parameters()
        #     # print(cam_param.extrinsic)


        #     key = cv2.waitKey(15)
        #     if key == 27:
        #         break




        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param = update_cam(cam_param, trans)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        capture_image(vis, outfilename=os.path.join(outimg_dir, 'img_{:06d}_cam1.png'.format(idx)))


        # vis.run()
        # capture_image(vis, outfilename=os.path.join(outimg_dir, 'img_{:06d}_cam1.png'.format(idx)))

        ### setup rendering cam, depth capture, segmentation capture    
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = trans2_dict[scene_name]
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        capture_image(vis, outfilename=os.path.join(outimg_dir, 'img_{:06d}_cam2.png'.format(idx)))



if __name__ == '__main__':

    import sys
    scenes = ['17DRP5sb8fy-bedroom', '17DRP5sb8fy-familyroomlounge', 
                '17DRP5sb8fy-livingroom', 'sKLMLpTHeUy-familyname_0_1', 
                'X7HyMhZNoso-livingroom_0_16', 'zsNo4HB9uLZ-bedroom0_0', 
                'zsNo4HB9uLZ-livingroom0_13']

    test_folder = sys.argv[1]

    for ss in scenes:

        gen_folder = os.path.join(test_folder, 'virtualcams', ss)
        outimg_dir = os.path.join(test_folder, 'virtualcams_img', ss)

        args = type('', (), {})()
        args.gen_folder = gen_folder
        args.outimg_dir = outimg_dir
        args.prox_dir = '/is/ps2/yzhang/workspaces/habitat-api/data/mp3d-rooms'
        args.model_folder = '/home/yzhang/body_models/VPoser/'
        args.num_pca_comps = 12
        args.gender = 'neutral'


        main(args)
