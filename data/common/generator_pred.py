import numpy as np


class ChunkedGenerator_pred:
    """ refined from https://github.com/facebookresearch/VideoPose3D
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:

    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False):

        # Build lineage info
        pairs = []
        self.saved_index = {}
        start_index = 0
   

            
        n_chunks = (poses_2d.shape[0] + chunk_length - 1) // chunk_length
        offset = (n_chunks * chunk_length - poses_2d.shape[0]) // 2
        bounds = np.arange(n_chunks + 1) * chunk_length - offset
        augment_vector = np.full(len(bounds - 1), False, dtype=bool)
        reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
        #keys=['pred' 'custom' '0']
        pairs += list(zip( bounds[:-1], bounds[1:], augment_vector,reverse_augment_vector))
        if reverse_aug:
            pairs += list(zip( bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
        if augment:
            if reverse_aug:
                pairs += list(zip( bounds[:-1], bounds[1:], ~augment_vector,~reverse_augment_vector))
            else:
                pairs += list(zip( bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))
        # for save key index
        end_index = start_index + poses_2d.shape[0]
        self.saved_index = [start_index,end_index]
        start_index = start_index + poses_2d.shape[0]

            
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d.shape[-2], poses_2d.shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.poses_3d = None
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state


    def get_batch(self, start_3d, end_3d, flip, reverse):

        #subject,action,cam_index = seq_i
        #seq_name = (subject,action,int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        # 2D poses
        seq_2d = self.poses_2d.copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                      'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            # Flip 2D keypoints
            self.batch_2d[ :, :, 0] *= -1
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :,
                                                                  self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        
        return self.batch_2d.copy()
        