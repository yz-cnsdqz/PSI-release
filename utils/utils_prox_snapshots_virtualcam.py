import os,sys
import os.path as osp
import cv2
import numpy as np
import json
import yaml
import open3d as o3d
import trimesh
import argparse
import matplotlib.pyplot as plt
import pdb

sys.path.append('/home/yzhang/workspaces/smpl-env-gen-3d-internal')


import torch
import torch.nn.functional as F
import pickle
import smplx
from human_body_prior.tools.model_loader import load_vposer
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.io as sio
import glob, copy
import random




###############################################################################
############################## helper functions ###############################
###############################################################################


def color_encoding(mesh):
    '''
    we use the color coding of Matterport3D
    '''
    def hex2rgb(hex_color_list):
        rgb_list = []
        for hex_color in hex_color_list:
            h = hex_color.lstrip('#')
            rgb = list(int(h[i:i+2], 16) for i in (0, 2, 4))
            rgb_list.append(rgb)

        return np.array(rgb_list)


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





def update_render_cam(cam_param,  trans):
    ### NOTE: trans is the trans relative to the world coordinate!!!

    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat

    return cam_param




def get_inner_normal(wall_dict, body_T):
    v1 = np.array(wall_dict['v1'])
    v2 = np.array(wall_dict['v2'])
    v3 = np.array(wall_dict['v3'])
    v4 = np.array(wall_dict['v4'])

    normal = np.cross(v2-v1, v3-v1)
    normal = normal / np.linalg.norm(normal)

    c = (v1+v2+v3+v4)/4.0

    if np.dot(normal, body_T - c) < 0:
        normal = -normal

    return normal, c



def get_new_cams(scene_name, s_min, s_max, body_T, body_R=None,
                    scene_grid_nodes=10,
                    discard_ratio=0.5):
    r_dict = room_dict_all[scene_name]

    if scene_name in ['MPH1Library','BasementSittingBooth', 'N0SittingBooth',
                        'N0Sofa','N3Library','N3OpenArea']:
        shift = 2.0
        s_min = s_min-shift
        s_max = s_max+shift
        # s_min -= shift
        # s_max += shift


    ## get plane inner normals
    normal_ceiling,cc = get_inner_normal(r_dict['ceiling'], body_T)
    normal_floor,cf = get_inner_normal(r_dict['floor'], body_T)
    normal_w1, cw1 = get_inner_normal(r_dict['wall_1'], body_T)
    normal_w2, cw2 = get_inner_normal(r_dict['wall_2'], body_T)
    normal_w3, cw3 = get_inner_normal(r_dict['wall_3'], body_T)
    normal_w4, cw4 = get_inner_normal(r_dict['wall_4'], body_T)
    r_center = (s_min + s_max) / 2.0


    ## get cam T via meshgrid. cam R is defined by C-body center
    scene_grid_nodes_xy = scene_grid_nodes
    scene_grid_nodes_z  = scene_grid_nodes_xy//3

    xy = np.linspace(s_min[:2], s_max[:2], num=scene_grid_nodes_xy)
    grid_x, grid_y, grid_z = np.meshgrid(xy[:,0], xy[:,1], np.linspace( body_T[-1],
                                                                     cc[-1],num=scene_grid_nodes_z)
                                        )

    ## for loop to select effecive cam pose
    cam_mat_list = []
    for i in range(1,scene_grid_nodes_xy-1):
        for j in range(1,scene_grid_nodes_xy-1):

            for k in range(1,scene_grid_nodes_z-1):

                ### get cam T
                cam_T_X = grid_x[i,j,k]
                cam_T_Y = grid_y[i,j,k]
                cam_T_Z = grid_z[i,j,k]
                cam_T = np.array([cam_T_X, cam_T_Y, cam_T_Z])

                ### get the cam R
                cam_z = body_T - cam_T
                cam_z = cam_z / np.linalg.norm(cam_z)
                cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
                cam_x = cam_x / np.linalg.norm(cam_x)
                cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
                cam_y = cam_y / np.linalg.norm(cam_y)
                rot_mat = np.stack([cam_x, -cam_y, cam_z], axis=1)

                cam_mat = np.eye(4)
                cam_T = cam_T + 0.5*np.random.randn()
                cam_mat[:-1,:-1] = rot_mat
                cam_mat[:-1,-1] = cam_T

                ### cam filter 1: distance to the body
                if (np.linalg.norm(cam_T - body_T) <= 1.65 or
                    np.linalg.norm(cam_T - body_T) >= 6.5):
                    continue


                ### cam filter 2: cam should be in the room
                if (np.dot(cam_T-cw1, normal_w1)<0 or
                    np.dot(cam_T-cw2, normal_w2)<0 or
                    np.dot(cam_T-cw3, normal_w3)<0 or
                    np.dot(cam_T-cw4, normal_w4)<0 or
                    np.dot(cam_T-cc, normal_ceiling)<0 or
                    np.dot(cam_T-cf, normal_floor)<0):
                    continue

                cam_mat_list.append(cam_mat)


    return cam_mat_list



def bodyparam_np2torch(param):

    torch_param = {}

    for key in param.keys():
        if key in ['body_pose','camera_rotation', 'camera_translation']:
            continue
        else:
            torch_param[key] = torch.tensor(param[key], dtype=torch.float32)

    return torch_param



def invert_transform(trans):
    _R = trans[:-1, :-1].T
    _T = -np.dot(_R, trans[:-1, -1])
    trans_inv = np.eye(4)
    trans_inv[:-1,:-1] = _R
    trans_inv[:-1, -1] = _T

    return trans_inv



def update_globalRT_for_smplx(body_params, smplx_model, vposer, trans_list, delta_T=None):
    '''
    input:
    body_params: dict, basically the input to the smplx model
    smplx_model: the model to generate smplx mesh, given body_params
    vposer: the model to generate body pose via vposer
    trans_list: coordinate transformation [4,4] mat

    Output:
    body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    if delta_T is None:
        torch_params = bodyparam_np2torch(body_params)
        if body_params['body_pose'].shape[-1] == 32:
            body_pose = vposer.decode(torch.tensor(body_params['body_pose']), 
                                        output_type='aa').view(1, -1)
        else:
            body_pose = torch.tensor(body_params['body_pose'])

        torch_params['transl'] = torch.zeros([1,3], dtype=torch.float32)
        torch_params['global_orient'] = torch.zeros([1,3], dtype=torch.float32)
        smplx_out = smplx_model(return_verts=True, body_pose=body_pose, **torch_params)
        delta_T = smplx_out.joints[0,0,:] # (3,)
        delta_T = delta_T.detach().cpu().numpy()


    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_params['global_orient'][0]
    body_R_mat = R.from_rotvec(body_R_angle).as_dcm() # to a [3,3] rotation mat
    body_T = body_params['transl'][0]
    body_mat = np.eye(4)
    body_mat[:-1,:-1] = body_R_mat
    body_mat[:-1, -1] = body_T + delta_T

    ### step (3): perform transformation, and decalib the delta shift
    trans_new_list = []

    for trans in trans_list:
        body_params_new = copy.deepcopy(body_params)
        body_mat_new = np.dot(trans, body_mat)
        body_R_new = R.from_dcm(body_mat_new[:-1,:-1]).as_rotvec()
        body_T_new = body_mat_new[:-1, -1]
        body_params_new['global_orient'] = body_R_new.reshape(1,3)
        body_params_new['transl'] = (body_T_new - delta_T).reshape(1,3)

        trans_new_list.append(body_params_new)


    return trans_new_list, delta_T






def data_preprocessing(img, modality, target_domain_size=[128, 128]):

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
    canvas = np.zeros(target_domain_size, dtype=float)

    # filter out unavailable values
    if modality == 'depth':
        img[img>6.0]=6.0
        # img[img==np.nan]=0.0

    if modality == 'seg':
        img[img>41] = 41

    ## rescale to [-1,1]
    # _img = (img-np.mean(img)) / np.std(img)
    _img = 2* img / np.max(img) - 1.0


    # put _img to the canvas
    if img_shape_o[0]>= img_shape_o[1]:
        factor = float(target_domain_size[0]) / img_shape_o[0]
        target_height = target_domain_size[0]
        target_width = int(img_shape_o[1] * factor) //2 *2

        # for depth map we use bilinear interpolation in resizing
        # for segmentation map we use bilinear interpolation as well.
        # note that float semantic label is not real in practice, but
        # helpful in our work

        target_size = [target_width,target_height]
        # img_resize = imresize(_img, output_shape=target_size,
        #                       anti_aliasing=True)

        img_resize = cv2.resize(_img, tuple(target_size) )

        na = target_width
        nb = target_domain_size[1]
        lower = (nb //2) - (na //2)
        upper = (nb //2) + (na //2)

        canvas[:, lower:upper] = img_resize


    else:
        factor = float(target_domain_size[1]) / img_shape_o[1]

        target_height = int(factor*img_shape_o[0]) //2 *2
        target_width = target_domain_size[1]

        target_size = [target_width,target_height]
        # img_resize = imresize(_img, output_shape=target_size)
        img_resize = cv2.resize(_img, tuple(target_size) )

        na = target_height
        nb = target_domain_size[0]
        lower = (nb //2) - (na //2)
        upper = (nb //2) + (na //2)

        canvas[lower:upper, :] = img_resize

    return canvas, factor



def is_body_occluded(body_param, cam_param, depth):

    global_T = body_param['transl'][0]

    cam_int = cam_param['intrinsic']

    fx = cam_int[0,0]
    fy = cam_int[1,1]
    h_, w_ = depth.shape


    ### if the human body center is occluded by the depth map, then return True

    body_c_x = int(global_T[0] * fx / global_T[2] + w_/2)
    body_c_y = int(global_T[1] * fy / global_T[2] + h_/2)

    # is the body in the image?
    if body_c_x <=10 or body_c_x > w_ - 10:
        return True

    if body_c_y <=10 or body_c_y > h_ - 10:
        return True


    d = 10
    lb_x = max(body_c_x-d, 0)
    lb_y = max(body_c_y-d, 0)
    ub_x = min(body_c_x+d, w_)
    ub_y = min(body_c_y+d, h_)

    # is the body torso occluded?
    if np.mean(depth[lb_y:ub_y, lb_x:ub_x]) <= global_T[-1]:
        return True



    return False







###############################################################################
############################### main function #################################
###############################################################################

def main(args):
    fitting_dir = args.fitting_dir
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = osp.join(fitting_dir, 'results')
    data_dir = args.data_dir
    cam2world_dir = osp.join(data_dir, 'cam2world')
    scene_dir = osp.join(data_dir, 'scenes_semantics')
    recording_dir = osp.join(data_dir, 'recordings', recording_name)
    scene_name = os.path.abspath(recording_dir).split("/")[-1].split("_")[0]


    ## setup the output folder
    output_folder = os.path.join('/mnt/hdd/PROX','snapshot_virtualcam_TNoise0.5')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    ### setup visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=270,visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face=True


    ### put the scene into the visualizer
    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '_withlabels.ply'))
    vis.add_geometry(scene)


    ## get scene 3D scene bounding box
    scene_o = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    scene_min = scene_o.get_min_bound() #[x_min, y_min, z_min]
    scene_max = scene_o.get_max_bound() #[x_max, y_max, z_max]
    # reduce the scene region furthermore, to avoid cams behind the window
    shift = 0.7
    scene_min = scene_min + shift
    scene_max = scene_max - shift


    ### get the real camera config
    trans_calib = np.eye(4)
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans_calib = np.array(json.load(f))


    ## put the body into the environment
    vposer_ckpt = osp.join(args.model_folder, 'vposer_v1_0')
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')

    model = smplx.create(args.model_folder, model_type='smplx',
                         gender='neutral', ext='npz',
                         num_pca_comps=12,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=False,
                         create_jaw_pose=False,
                         create_leye_pose=False,
                         create_reye_pose=False,
                         create_transl=True
                         )


    rec_count = -1
    sample_rate=15 # 0.5second
    for img_name in sorted(os.listdir(fitting_dir))[::sample_rate]:


        ## get humam body params
        filename =osp.join(fitting_dir, img_name, '000.pkl')
        print('frame: ' + filename)

        if not os.path.exists(filename):
            print('file does not exist. Continue')
            continue

        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            body_dict = pickle.load(f)

        if np.sum(np.isnan(body_dict['body_pose'])) > 0:
            continue

        rec_count = rec_count + 1
        ## save depth, semantics and render cam
        outname1 = os.path.join(output_folder,recording_name)
        if not os.path.exists(outname1):
            os.mkdir(outname1)



        ######################### then we obtain the virutal cam ################################

        ## find world coordinate of the human body in the current frame
        body_params_W_list, dT = update_globalRT_for_smplx(body_dict, model, vposer, [trans_calib])
        body_T_world = body_params_W_list[0]['transl'][0] + dT


        ## get virtual cams, and transform global_R and global_T to virtual cams
        new_cammat_ext_list0 = []
        new_cammat_ext_list0 = get_new_cams(scene_name, s_min=scene_min, s_max=scene_max,
                                            body_T=body_T_world)
        random.shuffle(new_cammat_ext_list0)
        new_cammat_ext_list = new_cammat_ext_list0[:30]
        
        print('--obtain {:d} cams'.format(len(new_cammat_ext_list)))

        new_cammat_list = [invert_transform(x) for x in new_cammat_ext_list]
        body_params_new_list, _ = update_globalRT_for_smplx(body_params_W_list[0],
                                                model, vposer, new_cammat_list,
                                                delta_T=dT)

        #### capture depth and seg in new cams
        for idx_cam, cam_ext in enumerate(new_cammat_ext_list):

            ### save filename
            outname = os.path.join(outname1, 'rec_frame{:06d}_cam{:06d}.mat'.format(rec_count, idx_cam))

            ## put the render cam to the real cam
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param = update_render_cam(cam_param, cam_ext)
            ctr.convert_from_pinhole_camera_parameters(cam_param)
            vis.poll_events()
            vis.update_renderer()

            ## get render cam parameters
            cam_dict = {}
            cam_dict['extrinsic'] = cam_param.extrinsic
            cam_dict['intrinsic'] = cam_param.intrinsic.intrinsic_matrix

            ## capture depth image
            depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
            _h = depth.shape[0]
            _w = depth.shape[1]
            depth0 = depth
            depth_canvas, scaling_factor = data_preprocessing(depth, 'depth')


            ### skip settings when the human body is severely occluded.
            body_is_occluded = is_body_occluded(body_params_new_list[idx_cam], cam_dict, depth)

            if body_is_occluded:
                print('-- body is occluded or not in the scene at current view.')
                continue

            ## capture semantics
            seg = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            verid = np.mean(seg*255/5.0,axis=-1) #.astype(int)
            seg0 = verid
            # verid = cv2.resize(verid, (_w//factor, _h//factor))
            seg_canvas, _ = data_preprocessing(verid, 'seg')

            # pdb.set_trace()

            ## save file to disk
            ot_dict={}
            ot_dict['scaling_factor']=scaling_factor
            ot_dict['depth']=depth_canvas
            ot_dict['depth0']=depth0
            ot_dict['seg0']=seg0
            ot_dict['seg'] = seg_canvas
            ot_dict['cam'] = cam_dict
            ot_dict['body'] = body_params_new_list[idx_cam]
            sio.savemat(outname, ot_dict)


    vis.destroy_window()



class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':

    fitting_dir_list = glob.glob('/mnt/hdd/PROX/PROXD/*')
    data_dir = '/mnt/hdd/PROX'
    model_folder  = '/home/yzhang/body_models/VPoser/'
    args= {}


    ### read dict of virutal wall, ceiling and floor
    with open('/mnt/hdd/PROX/PROXE_box_verts.json','r') as f:
        room_dict_all = json.load(f)


    for fitting_dir in fitting_dir_list:
        print('-- process {}'.format(fitting_dir))
        args['fitting_dir'] = fitting_dir
        args['data_dir'] = data_dir
        args['model_folder'] = model_folder
        main(Struct(**args))
