# Mostly based on the code written by Tinghui Zhou:
# https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py

from __future__ import division
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os


def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, dump_root, data_loader):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if not example:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(dump_root, example['folder_name'])

    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.png' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))


def gen_data_kitti(dataset_dir, dump_root, img_height=128, img_width=416,
                   seq_length=3):
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)

    from .kitti_raw_loader import kitti_raw_loader
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'
    data_loader = kitti_raw_loader(dataset_dir,
                                   split='eigen',
                                   img_height=img_height,
                                   img_width=img_width,
                                   seq_length=seq_length,
                                   remove_static=True)

    Parallel(n_jobs=8)(delayed(dump_example)(n, dump_root, data_loader)
                       for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(8964)
    if dump_root.endswith('/'):
        dump_root = dump_root[:-1]
    subfolders = os.listdir(dump_root)
    with open(os.path.join(dump_root, 'train.txt'), 'w') as tf:
        with open(os.path.join(dump_root, 'val.txt'), 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(dump_root, s, '*.png'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))
