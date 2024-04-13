"""
refined from https://github.com/facebookresearch/VideoPose3D
"""
import numpy as np
import copy
from data.common.skeleton import Skeleton
from data.common.mocap_dataset import MocapDataset
from data.common.camera import normalize_screen_coordinates

synthetic_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8,  11,  12,  8,  14,  15],
       joints_left=[4, 5, 6, 11, 12, 13],
       joints_right=[1, 2, 3, 14, 15, 16])

subjects=['S9']



cam_map = {
    'Camera_0': 0,
}

import sys
sys.path.append('../')
path_synth = '../Synthetic_H3.6M' 
from data.common.utils import wrap
from data.common.quaternion import qrot, qinverse

class TestDataset(MocapDataset):
    def __init__(self, path, opt,remove_static_joints=True):
        super().__init__(fps=50, skeleton=synthetic_skeleton)
                
        cameras_extrinsic_params_dict = {}
        for subject in subjects: 
            
            file_path = path_synth + '/' + subject + '/Cameras/camera_params_'+ subject+'.npz'
            # Load serialized dataset
            data_int = np.load(file_path, allow_pickle=True)['intrinsic_params']
            data_ext = np.load(file_path, allow_pickle=True)['extrinsic_params'].item()[subject]
                       
            cameras_intrinsic_params = []
            cameras_extrinsic_params = []  
            # List to hold individual camera extrinsic parameter dictionaries for each subject
            for camera in data_int:
                
                camera_int_params = {
                    'id': cam_map[camera['id']],
                    'center': camera['center'],
                    'focal_length': camera['focal_length'],
                    'radial_distortion': camera['radial_distortion'],
                    'tangential_distortion': camera['tangential_distortion'],
                    'res_w': camera['res_w'],
                    'res_h': camera['res_h'],
                    'azimuth': camera['azimuth']  # Only used for visualization 70?
                }
                cameras_intrinsic_params.append(camera_int_params)
            
            for camera in data_ext:
                T=camera['translation']
                R=camera['orientation']
                q_inv=wrap(qinverse, np.array(R))
                
                extrinsic_params={
                    'orientation': q_inv,
                    'translation': -wrap(qrot, q_inv, np.array(T))
                }

                cameras_extrinsic_params.append(extrinsic_params)

            cameras_extrinsic_params_dict={
                subject:cameras_extrinsic_params
            }
           
             
            self._cameras = copy.deepcopy(cameras_extrinsic_params_dict)
          
               
            
            for cameras in self._cameras.values():
                for i, cam in enumerate(cameras):
                    cam.update(cameras_intrinsic_params[i])
                    for k, v in cam.items():
                        if k not in ['id', 'res_w', 'res_h']:
                            cam[k] = np.array(v, dtype='float32')

                    # Normalize camera frame
                    cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                    cam['focal_length'] = cam['focal_length']/cam['res_w']*2

                    # Add intrinsic parameters vector
                    cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                       cam['center'],
                                                       cam['radial_distortion'],
                                                       cam['tangential_distortion']))

        # Load serialized dataset
        
        data = np.load(path,allow_pickle=True)['positions_3d'].item()
        
        self._data = {}
        for subject, actions in data.items():
           
            self._data[subject] = {}
            for action_name, positions in actions.items():
                
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }
        


            
    def supports_semi_supervised(self):
        return True