# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains common utilities and functions. Based on struct2depth"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl import logging
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

gfile = tf.gfile

CMAP_DEFAULT = 'plasma'
# Defines the cropping that is applied to the Cityscapes dataset with respect to
# the original raw input resolution.
CITYSCAPES_CROP = [256, 768, 192, 1856]


def crop_cityscapes(im, resize=None):
    ymin, ymax, xmin, xmax = CITYSCAPES_CROP
    im = im[ymin:ymax, xmin:xmax]
    if resize is not None:
        im = cv2.resize(im, resize)
    return im


def gray2rgb(im, cmap=CMAP_DEFAULT):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def load_image(img_file, resize=None, interpolation='linear', seg_image=False, transpose=False):
    """Load image from disk. Output value range: [0,1]."""
    # im_data = np.fromstring(gfile.Open(img_file).read(), np.uint8)  # original
    # Ref: https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8
    # -codec-cant-decode-byte-0xff-in-position-0-in/48556203
    im_data = np.fromstring(gfile.Open(img_file, 'rb').read(), np.uint8)
    im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if transpose:
        im = np.transpose(im, (1, 0, 2))

    if resize and resize != im.shape[:2]:
        ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
        im = cv2.resize(im, resize, interpolation=ip)
    if seg_image:
        # For segmented image, load as uint8
        return np.array(im, dtype=np.uint8)
    else:
        return np.array(im, dtype=np.float32) / 255.0


def save_image(img_file, im, file_extension):
    """Save image from disk. Expected input value range: [0,1]."""
    im = (im * 255.0).astype(np.uint8)
    with gfile.Open(img_file, 'w') as f:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        _, im_data = cv2.imencode('.%s' % file_extension, im)
        f.write(im_data.tostring())


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap=CMAP_DEFAULT):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.

    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    return disp


def get_seq_start_end(target_index, seq_length, sample_every=1):
    """Returns absolute seq start and end indices for a given target frame."""
    half_offset = int((seq_length - 1) / 2) * sample_every
    end_index = target_index + half_offset
    start_index = end_index - (seq_length - 1) * sample_every
    return start_index, end_index


def get_seq_middle(seq_length):
    """Returns relative index for the middle frame in sequence."""
    half_offset = int((seq_length - 1) / 2)
    return seq_length - 1 - half_offset


def info(obj):
    """Return info on shape and dtype of a numpy array or TensorFlow tensor."""
    if obj is None:
        return 'None.'
    elif isinstance(obj, list):
        if obj:
            return 'List of %d... %s' % (len(obj), info(obj[0]))
        else:
            return 'Empty list.'
    elif isinstance(obj, tuple):
        if obj:
            return 'Tuple of %d... %s' % (len(obj), info(obj[0]))
        else:
            return 'Empty tuple.'
    else:
        if is_a_numpy_array(obj):
            return 'Array with shape: %s, dtype: %s' % (obj.shape, obj.dtype)
        else:
            return str(obj)


def is_a_numpy_array(obj):
    """Returns true if obj is a numpy array."""
    return type(obj).__module__ == np.__name__


def count_parameters(also_print=True):
    """Count the number of parameters in the model.

    Args:
      also_print: Boolean.  If True also print the numbers.

    Returns:
      The total number of parameters.
    """
    total = 0
    if also_print:
        logging.info('Model Parameters:')
    for (_, v) in get_vars_to_save_and_restore().items():
        shape = v.get_shape()
        if also_print:
            logging.info('%s %s: %s', v.op.name, shape,
                         format_number(shape.num_elements()))
        total += shape.num_elements()
    if also_print:
        logging.info('Total: %s', format_number(total))
    return total


def get_vars_to_save_and_restore(ckpt=None):
    """Returns list of variables that should be saved/restored.

    Args:
      ckpt: Path to existing checkpoint.  If present, returns only the subset of
          variables that exist in given checkpoint.

    Returns:
      List of all variables that need to be saved/restored.
    """
    model_vars = tf.trainable_variables()
    # Add batchnorm variables.
    bn_vars = [v for v in tf.global_variables()
               if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name or
               'mu' in v.op.name or 'sigma' in v.op.name or
               'global_scale_var' in v.op.name]
    model_vars.extend(bn_vars)
    model_vars = sorted(model_vars, key=lambda x: x.op.name)
    mapping = {}
    if ckpt is not None:
        ckpt_var = tf.contrib.framework.list_variables(ckpt)
        ckpt_var_names = [name for (name, unused_shape) in ckpt_var]
        ckpt_var_shapes = [shape for (unused_name, shape) in ckpt_var]
        not_loaded = list(ckpt_var_names)
        for v in model_vars:
            if v.op.name not in ckpt_var_names:
                # For backward compatibility, try additional matching.
                v_additional_name = v.op.name.replace('egomotion_prediction/', '')
                if v_additional_name in ckpt_var_names:
                    # Check if shapes match.
                    ind = ckpt_var_names.index(v_additional_name)
                    if ckpt_var_shapes[ind] == v.get_shape():
                        mapping[v_additional_name] = v
                        not_loaded.remove(v_additional_name)
                        continue
                    else:
                        logging.warning('Shape mismatch, will not restore %s.', v.op.name)
                logging.warning('Did not find var %s in checkpoint: %s', v.op.name,
                                os.path.basename(ckpt))
            else:
                # Check if shapes match.
                ind = ckpt_var_names.index(v.op.name)
                if ckpt_var_shapes[ind] == v.get_shape():
                    mapping[v.op.name] = v
                    not_loaded.remove(v.op.name)
                else:
                    logging.warning('Shape mismatch, will not restore %s.', v.op.name)
        if not_loaded:
            logging.warning('The following variables in the checkpoint were not loaded:')
            for varname_not_loaded in not_loaded:
                logging.info('%s', varname_not_loaded)
    else:  # just get model vars.
        for v in model_vars:
            mapping[v.op.name] = v
    return mapping


def get_imagenet_vars_to_restore(imagenet_ckpt):
    """Returns dict of variables to restore from ImageNet-checkpoint."""
    vars_to_restore_imagenet = {}
    ckpt_var_names = tf.contrib.framework.list_variables(imagenet_ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
    model_vars = tf.global_variables()

    for v in model_vars:
        if 'global_step' in v.op.name:
            continue
        mvname_noprefix = v.op.name.replace('depth_prediction/', '')
        mvname_noprefix = mvname_noprefix.replace('moving_mean', 'mu')
        mvname_noprefix = mvname_noprefix.replace('moving_variance', 'sigma')
        if mvname_noprefix in ckpt_var_names:
            vars_to_restore_imagenet[mvname_noprefix] = v
        else:
            logging.info('The following variable will not be restored from '
                         'pretrained ImageNet-checkpoint: %s', mvname_noprefix)
    return vars_to_restore_imagenet


def format_number(n):
    """Formats number with thousands commas."""
    # locale.setlocale(locale.LC_ALL, 'en_US')  # commented by me
    # return locale.format('%d', n, grouping=True)
    return n


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def read_text_lines(filepath):
    with tf.gfile.Open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def save_flags(FLAGS, save_path):
    import json
    check_path(save_path)
    save_path = os.path.join(save_path, 'flags.json')
    with open(save_path, 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f, indent=4, sort_keys=False)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_command(save_path):
    check_path(save_path)
    import sys
    command = sys.argv
    save_file = os.path.join(save_path, 'command.txt')
    with open(save_file, 'w') as f:
        f.write(' '.join(command))


def make_intrinsics_matrix(fx, fy, cx, cy):
    r1 = np.stack([fx, 0, cx])
    r2 = np.stack([0, fy, cy])
    r3 = np.array([0., 0., 1.])
    intrinsics = np.stack([r1, r2, r3])
    return intrinsics


def get_multi_scale_intrinsics(intrinsics, num_scales):
    """Returns multiple intrinsic matrices for different scales."""
    intrinsics_multi_scale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[0, 0] / (2 ** s)
        fy = intrinsics[1, 1] / (2 ** s)
        cx = intrinsics[0, 2] / (2 ** s)
        cy = intrinsics[1, 2] / (2 ** s)
        intrinsics_multi_scale.append(make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_multi_scale = np.stack(intrinsics_multi_scale)  # [num_scales, 3, 3]
    return intrinsics_multi_scale


def pack_pred_depths(pred_dir, test_file):
    """Pack depth predictions as a single .npy file"""
    test_images = read_text_lines(test_file)

    save_name = 'pred_depth.npy'
    output_file = os.path.join(pred_dir, save_name)

    img_height = 128
    img_width = 416

    all_pred = np.zeros((len(test_images), img_height, img_width))

    for i, img_path in enumerate(test_images):
        npy_path = os.path.join(pred_dir, img_path.replace('png', 'npy'))
        depth = np.load(npy_path)
        all_pred[i] = np.squeeze(depth)

    np.save(output_file, all_pred)


# Depth evaluation utils
# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# EIGEN split


def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0
    for filename in files:
        filename = filename.split()[0]
        splits = filename.split('/')
        #         camera_id = filename[-1]   # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        file_root = '{}/{}'

        im = filename
        vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

        if os.path.isfile(data_root + im):
            gt_files.append(data_root + vel)
            gt_calib.append(data_root + date + '/')
            im_sizes.append(cv2.imread(data_root + im).shape[:2])
            im_files.append(data_root + im)
            cams.append(2)
        else:
            num_probs += 1
            print('{} missing'.format(data_root + im))
    # print(num_probs, 'files missing')

    return gt_files, gt_calib, im_sizes, im_files, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir, cam=2):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P3_rect = cam2cam['P_rect_03'].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros(im_shape)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
