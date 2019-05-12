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

"""Geometry utilities for projecting frames based on depth and motion.

Modified from Spatial Transformer Networks:
https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def inverse_warp(img, depth, egomotion_mat, intrinsic_mat,
                 intrinsic_mat_inv):
    """Inverse warp a source image to the target image plane.

    Args:
      img: The source image (to sample pixels from) -- [B, H, W, 3].
      depth: Depth map of the target image -- [B, H, W].
      egomotion_mat: Matrix defining egomotion transform -- [B, 4, 4].
      intrinsic_mat: Camera intrinsic matrix -- [B, 3, 3].
      intrinsic_mat_inv: Inverse of the intrinsic matrix -- [B, 3, 3].
    Returns:
      Projected source image
    """
    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])  # [B, 1, H*W]
    grid = _meshgrid_abs(img_height, img_width)  # [3, H*W]
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])  # [B, 3, H*W]
    cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)  # [B, 3, H*W]
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)  # [B, 4, H*W]

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsic_mat_hom = tf.concat(
        [intrinsic_mat, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)  # [B, 4, 4]
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, egomotion_mat)  # [B, 4, 4]
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                     proj_target_cam_to_source_pixel)  # [B, 2, H*W]
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                     [batch_size, 2, img_height, img_width])  # [B, 2, H, W]
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])  # [B, H, W, 2]
    projected_img, mask = spatial_transformer(img, source_pixel_coords)
    return projected_img, mask


def region_deformer(img, trans_params, residual=True):
    """Region deformer to warp the source image to target
    Args:
        img: [B, H, W, 3]
        trans_params: [B, 16, 2], for bicubic function
        residual: learn residual transform or direct transform
    Returns:
        transformed image, mask
    """
    dims = tf.shape(img)
    batch_size, height, width = dims[0], dims[1], dims[2]
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))  # [H, W]
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))  # [H, W]

    x_t = tf.tile(tf.expand_dims(x_t, axis=0), [batch_size, 1, 1])  # [B, H, W]
    y_t = tf.tile(tf.expand_dims(y_t, axis=0), [batch_size, 1, 1])  # [B, H, W]

    x_t = (x_t + 1.0) * 0.5  # in [0, 1]
    y_t = (y_t + 1.0) * 0.5

    x_t_square = tf.square(x_t)
    x_t_pow3 = tf.pow(x_t, 3)

    y_t_square = tf.square(y_t)
    y_t_pow3 = tf.pow(y_t, 3)
    ones = tf.ones_like(x_t)

    grid_all = tf.stack([ones, x_t, x_t_square, x_t_pow3, y_t, x_t * y_t, x_t_square * y_t, x_t_pow3 * y_t,
                         y_t_square, x_t * y_t_square, x_t_square * y_t_square, x_t_pow3 * y_t_square,
                         y_t_pow3, x_t * y_t_pow3, x_t_square * y_t_pow3, x_t_pow3 * y_t_pow3],
                        axis=3)  # [B, H, W, 16]
    trans_grid = tf.matmul(tf.reshape(grid_all, [batch_size, -1, 16]), trans_params)  # [B, H*W, 2]

    source_coords = tf.reshape(trans_grid, [batch_size, height, width, 2])  # [B, H, W, 2]

    source_coords = tf.stack([source_coords[:, :, :, 0] * tf.cast(width - 1, tf.float32),
                              source_coords[:, :, :, 1] * tf.cast(height - 1, tf.float32)],
                             axis=3)  # [B, H, W, 2]

    if residual:  # learning residual transform
        ori_grid = tf.stack([x_t * tf.cast(width - 1, tf.float32),
                             y_t * tf.cast(height - 1, tf.float32)],
                            axis=3)  # [B, H, W, 2]

        source_coords = ori_grid + source_coords

    transformed_img, mask = spatial_transformer(img, source_coords)

    return transformed_img, mask


def get_transform_mat(egomotion_vecs, i, j, use_axis_angle=False):
    """Returns a transform matrix defining the transform from frame i to j."""
    egomotion_transforms = []
    batchsize = tf.shape(egomotion_vecs)[0]
    if i == j:
        return tf.tile(tf.expand_dims(tf.eye(4, 4), axis=0), [batchsize, 1, 1])
    for k in range(min(i, j), max(i, j)):
        if use_axis_angle:
            transform_matrix = _egomotion_axisangle2mat(egomotion_vecs[:, k, :], batchsize)
        else:
            transform_matrix = _egomotion_vec2mat(egomotion_vecs[:, k, :], batchsize)
        if i > j:  # Going back in sequence, need to invert egomotion.
            egomotion_transforms.insert(0, tf.matrix_inverse(transform_matrix))
        else:  # Going forward in sequence
            egomotion_transforms.append(transform_matrix)

    # Multiply all matrices.
    egomotion_mat = egomotion_transforms[0]
    for i in range(1, len(egomotion_transforms)):
        egomotion_mat = tf.matmul(egomotion_mat, egomotion_transforms[i])
    return egomotion_mat


def get_region_deformer_params(trans_params, i, j):
    """Return a transform matrix defining the transform from frame i to j"""
    assert abs(i - j) < 2  # only consider 0 to 1, 1 to 1, 2 to 1
    assert j == 1
    dims = tf.shape(trans_params)  # [B, 2, trans_params_size]
    batch_size, trans_params_size = dims[0], dims[2]
    trans_mat_size = tf.cast(trans_params_size / 2, tf.int32)
    if i == j:
        raise ValueError('i == j should be skipped')

    idx = 0 if i == 0 else 1
    trans_mat_stack = tf.reshape(trans_params[:, idx], [batch_size, trans_mat_size, 2])  # [B, 16, 2]
    return trans_mat_stack


def compute_rigid_flow(depth, egomotion_mat, intrinsic_mat, intrinsic_mat_inv):
    """Rigid flow caused by camera motion

    Args:
      depth: Depth map of the target image -- [B, H, W].
      egomotion_mat: Matrix defining egomotion transform -- [B, 4, 4].
      intrinsic_mat: Camera intrinsic matrix -- [B, 3, 3].
      intrinsic_mat_inv: Inverse of the intrinsic matrix -- [B, 3, 3].
    Returns:
      Projected source image
    """
    dims = tf.shape(depth)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])  # [B, 1, H*W]
    grid = _meshgrid_abs(img_height, img_width)  # [3, H*W]
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])  # [B, 3, H*W]

    pixel_coords = tf.reshape(grid, [batch_size, 3, img_height, img_width])  # [B, 3, H, W]
    target_pixel_coords = tf.transpose(pixel_coords[:, :2], [0, 2, 3, 1])  # [B, H, W, 2]

    cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)  # [B, 3, H*W]
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)  # [B, 4, H*W]

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsic_mat_hom = tf.concat(
        [intrinsic_mat, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)  # [B, 4, 4]
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, egomotion_mat)  # [B, 4, 4]
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                     proj_target_cam_to_source_pixel)  # [B, 2, H*W]
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                     [batch_size, 2, img_height, img_width])  # [B, 2, H, W]
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])  # [B, H, W, 2]

    rigid_flow = source_pixel_coords - target_pixel_coords

    return rigid_flow


def compute_residual_flow(img, trans_params):
    """Compute residual flow from learned transform parameters
    Args:
        img: [B, H, W, 3], unused, only use shape info
        trans_params: [B, 16, 2], for bicubic function
    Returns:
        residual flow
    """
    dims = tf.shape(img)
    batch_size, height, width = dims[0], dims[1], dims[2]
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))  # [H, W]
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))  # [H, W]

    x_t = tf.tile(tf.expand_dims(x_t, axis=0), [batch_size, 1, 1])  # [B, H, W]
    y_t = tf.tile(tf.expand_dims(y_t, axis=0), [batch_size, 1, 1])  # [B, H, W]

    x_t = (x_t + 1.0) * 0.5  # in [0, 1]
    y_t = (y_t + 1.0) * 0.5

    x_t_square = tf.square(x_t)
    x_t_pow3 = tf.pow(x_t, 3)

    y_t_square = tf.square(y_t)
    y_t_pow3 = tf.pow(y_t, 3)
    ones = tf.ones_like(x_t)

    grid_all = tf.stack([ones, x_t, x_t_square, x_t_pow3, y_t, x_t * y_t, x_t_square * y_t, x_t_pow3 * y_t,
                         y_t_square, x_t * y_t_square, x_t_square * y_t_square, x_t_pow3 * y_t_square,
                         y_t_pow3, x_t * y_t_pow3, x_t_square * y_t_pow3, x_t_pow3 * y_t_pow3],
                        axis=3)  # [B, H, W, 16]
    trans_grid = tf.matmul(tf.reshape(grid_all, [batch_size, -1, 16]), trans_params)  # [B, H*W, 2]

    residual_flow = tf.reshape(trans_grid, [batch_size, height, width, 2])  # [B, H, W, 2]

    # Scale to pixel metric
    residual_flow = tf.stack([residual_flow[:, :, :, 0] * tf.cast(width - 1, tf.float32),
                              residual_flow[:, :, :, 1] * tf.cast(height - 1, tf.float32)],
                             axis=3)  # [B, H, W, 2]

    return residual_flow


def rigid_residual_flow_warp(img, rigid_flow, residual_flow):
    """Warp by rigid flow plus residual flow directly for moving objects
    Args:
        img: [B, H, W, 3], where to sample pixels
        rigid_flow: [B, H, W, 2], computed by estimated depth and egomotion
        residual_flow: [B, H, W, 2], computed by region deformer
    Returns:
        warped_img: [B, H, W, 3]
        mask: [B, H, W, 1]
    """
    full_flow = rigid_flow + residual_flow  # [B, H, W, 2]
    dim = tf.shape(img)
    batch_size, img_height, img_width = dim[0], dim[1], dim[2]
    grid = _meshgrid_abs(img_height, img_width)  # [3, H*W]
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])  # [B, 3, H*W]

    pixel_coords = tf.reshape(grid, [batch_size, 3, img_height, img_width])  # [B, 3, H, W]
    target_pixel_coords = tf.transpose(pixel_coords[:, :2], [0, 2, 3, 1])  # [B, H, W, 2]
    source_pixel_coords = target_pixel_coords + full_flow

    warped_img, mask = spatial_transformer(img, source_pixel_coords)

    return warped_img, mask


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth
    return cam_coords


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = tf.matmul(proj_c2p, cam_coords)  # [B, 4, H*W], [B, 4, 4] * [B, 4, H*W]
    x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
    y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
    z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
    # Not tested if adding a small number is necessary
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = tf.concat([x_norm, y_norm], axis=1)  # [B, 2, H*W]
    return pixel_coords


def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    # x_t 's column has same elements
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))  # [H, W], [H, 1] * [1, W]
    # y_t 's row has same elements
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))  # [H, W]
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    x_t_flat = tf.reshape(x_t, (1, -1))  # [0, 1, ..., W-1, 0, 1, ...]
    y_t_flat = tf.reshape(y_t, (1, -1))  # [0, 0, ..., 0, 1, 1, ...]
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)  # [3, H*W]
    return grid


def _euler2mat(z, y, x):
    """Converts euler angles to rotation matrix.

     From:
     https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

     TODO: Remove the dimension for 'N' (deprecated for converting all source
     poses altogether).

    Args:
      z: rotation angle along z axis (in radians) -- size = [B, n]
      y: rotation angle along y axis (in radians) -- size = [B, n]
      x: rotation angle along x axis (in radians) -- size = [B, n]

    Returns:
      Rotation matrix corresponding to the euler angles, with shape [B, n, 3, 3].
    """
    batch_size = tf.shape(z)[0]
    n = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([batch_size, n, 1, 1])
    ones = tf.ones([batch_size, n, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    return tf.matmul(tf.matmul(xmat, ymat), zmat)


def _axis_angle2mat(rvecs):
    """
    Code: https://github.com/blzq/tf_rodrigues/blob/master/rodrigues.py
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    # tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keep_dims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([zero, -u[:, 2], u[:, 1]], axis=1)  # row 1
    Ks_2 = tf.stack([u[:, 2], zero, -u[:, 0]], axis=1)  # row 2
    Ks_3 = tf.stack([-u[:, 1], u[:, 0], zero], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero,
                    tf.eye(3, batch_shape=[batch_size]), Rs)


def _egomotion_axisangle2mat(vec, batch_size):
    """Converts axis-angle representation to transformation matrix.

    Args:
      vec: 6DoF parameters [tx, ty, tz, rx, ry, rz] -- [B, 6].
      batch_size: Batch size.

    Returns:
      A transformation matrix -- [B, 4, 4].
    """
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)  # [B, 3, 1]

    axis_angle = tf.slice(vec, [0, 3], [-1, 3])  # [B, 3]
    rot_mat = _axis_angle2mat(axis_angle)  # [B, 3, 3]

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def _egomotion_vec2mat(vec, batch_size):
    """Converts 6DoF transform vector to transformation matrix.

    Args:
      vec: 6DoF parameters [tx, ty, tz, rx, ry, rz] -- [B, 6].
          if use 6d rotation representation, [B, 9], [tx, ty, tz, 6d]
      batch_size: Batch size.

    Returns:
      A transformation matrix -- [B, 4, 4].
    """
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)  # [B, 3, 1]

    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = _euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])  # [B, 3, 3]

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def _bilinear_sampler(im, x, y, name='blinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.

    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.

    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.

    Args:
      im: Batch of images with shape [B, h, w, channels].
      x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
      y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
      name: Name scope for ops.

    Returns:
      Sampled image with shape [B, h, w, channels].
      Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
    """
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # Constants.
        batch_size = tf.shape(im)[0]
        _, height, width, channels = im.get_shape().as_list()

        x = tf.to_float(x)
        y = tf.to_float(y)
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Compute the coordinates of the 4 pixels to sample from.
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        mask = tf.logical_and(
            tf.logical_and(x0 >= zero, x1 <= max_x),
            tf.logical_and(y0 >= zero, y1 <= max_y))
        mask = tf.to_float(mask)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height

        # Create base index.
        base = tf.range(batch_size) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, height * width])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore channels dim.
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        x1_f = tf.to_float(x1)
        y1_f = tf.to_float(y1)

        # And finally calculate interpolated values.
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
        wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
        mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))
        return output, mask


def spatial_transformer(img, coords):
    """A wrapper over bilinear_sampler(), taking absolute coords as input."""
    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = _bilinear_sampler(img, px, py)
    return output_img, mask
