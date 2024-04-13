"""
fuse training and testing

"""
import torch.utils.data as data

from data.common.camera import *
from data.common.utils import deterministic_random
from data.common.generator_pred import ChunkedGenerator_pred



class Fusion_pred(data.Dataset):
    def __init__(self, opt, dataset, root_path):
        self.data_type = opt.dataset
        self.keypoints_name = opt.keypoints
        self.root_path = root_path
        self.train = False
        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        # self.rescale = opt.rescale
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
    
        self.keypoints = np.load(opt.root_path + 'VP3D_predictions' + '.npy',allow_pickle=True)
        keypoints_symmetry = [[4, 5, 6, 11, 12, 13],[1, 2, 3, 14, 15, 16],]
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())

        self.generator = ChunkedGenerator_pred(opt.batchSize // opt.stride, self.keypoints,
                                          pad=self.pad, augment=False, kps_left=self.kps_left,
                                          kps_right=self.kps_right, joints_left=self.joints_left,
                                          joints_right=self.joints_right)
        self.key_index = self.generator.saved_index
        print('INFO: Testing on {} frames'.format(self.generator.num_frames()))
        

  
    def __len__(self):
        "Figure our how many sequences we have"

        return len(self.generator.pairs)

    def __getitem__(self, index):
       
        start_3d, end_3d, flip, reverse = self.generator.pairs[index]
       
        input_2D = self.generator.get_batch( start_3d, end_3d, flip, reverse)
        '''
        if self.train == False and self.test_aug:

            input_2D_aug = self.generator.get_batch( start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        '''
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)
       

        return input_2D_update








