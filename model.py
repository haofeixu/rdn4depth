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

"""Build model for inference or training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf

import nets
import project
import reader
import util

gfile = tf.gfile
slim = tf.contrib.slim

NUM_SCALES = 4


class Model(object):
    """Model code based on struct2depth."""

    def __init__(self,
                 data_dir=None,
                 file_extension='png',
                 is_training=True,
                 learning_rate=0.0002,
                 beta1=0.9,
                 reconstr_weight=0.85,
                 smooth_weight=0.05,
                 object_depth_weight=0.0,
                 object_depth_threshold=0.01,
                 exclude_object_mask=True,
                 stop_egomotion_gradient=True,
                 ssim_weight=0.15,
                 batch_size=4,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 architecture=nets.RESNET,
                 imagenet_norm=True,
                 weight_reg=0.05,
                 exhaustive_mode=False,
                 random_scale_crop=False,
                 flipping_mode=reader.FLIP_RANDOM,
                 random_color=True,
                 depth_upsampling=True,
                 depth_normalization=True,
                 compute_minimum_loss=True,
                 use_skip=True,
                 use_axis_angle=False,
                 joint_encoder=True,
                 build_sum=True,
                 shuffle=True,
                 input_file='train',
                 handle_motion=False,
                 equal_weighting=False,
                 same_trans_rot_scaling=True,
                 residual_deformer=True,
                 seg_align_type='null',
                 use_rigid_residual_flow=True,
                 region_deformer_scaling=1.0):
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.reconstr_weight = reconstr_weight
        self.smooth_weight = smooth_weight
        self.ssim_weight = ssim_weight
        self.object_depth_weight = object_depth_weight
        self.object_depth_threshold = object_depth_threshold
        self.exclude_object_mask = exclude_object_mask
        self.beta1 = beta1
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.architecture = architecture
        self.imagenet_norm = imagenet_norm
        self.weight_reg = weight_reg
        self.exhaustive_mode = exhaustive_mode
        self.random_scale_crop = random_scale_crop
        self.flipping_mode = flipping_mode
        self.random_color = random_color
        self.depth_upsampling = depth_upsampling
        self.depth_normalization = depth_normalization
        self.compute_minimum_loss = compute_minimum_loss
        self.use_skip = use_skip
        self.joint_encoder = joint_encoder
        self.build_sum = build_sum
        self.shuffle = shuffle
        self.input_file = input_file
        self.handle_motion = handle_motion
        self.equal_weighting = equal_weighting
        self.same_trans_rot_scaling = same_trans_rot_scaling
        self.residual_deformer = residual_deformer
        self.seg_align_type = seg_align_type
        self.use_rigid_residual_flow = use_rigid_residual_flow
        self.region_deformer_scaling = region_deformer_scaling
        self.stop_egomotion_gradient = stop_egomotion_gradient
        self.use_axis_angle = use_axis_angle

        self.trans_params_size = 32  # parameters of the bicubic function

        logging.info('data_dir: %s', data_dir)
        logging.info('file_extension: %s', file_extension)
        logging.info('is_training: %s', is_training)
        logging.info('learning_rate: %s', learning_rate)
        logging.info('reconstr_weight: %s', reconstr_weight)
        logging.info('smooth_weight: %s', smooth_weight)
        logging.info('ssim_weight: %s', ssim_weight)
        logging.info('beta1: %s', beta1)
        logging.info('batch_size: %s', batch_size)
        logging.info('img_height: %s', img_height)
        logging.info('img_width: %s', img_width)
        logging.info('seq_length: %s', seq_length)
        logging.info('architecture: %s', architecture)
        logging.info('imagenet_norm: %s', imagenet_norm)
        logging.info('weight_reg: %s', weight_reg)
        logging.info('exhaustive_mode: %s', exhaustive_mode)
        logging.info('random_scale_crop: %s', random_scale_crop)
        logging.info('flipping_mode: %s', flipping_mode)
        logging.info('random_color: %s', random_color)
        logging.info('depth_upsampling: %s', depth_upsampling)
        logging.info('depth_normalization: %s', depth_normalization)
        logging.info('compute_minimum_loss: %s', compute_minimum_loss)
        logging.info('use_skip: %s', use_skip)
        logging.info('joint_encoder: %s', joint_encoder)
        logging.info('build_sum: %s', build_sum)
        logging.info('shuffle: %s', shuffle)
        logging.info('input_file: %s', input_file)
        logging.info('handle_motion: %s', handle_motion)
        logging.info('equal_weighting: %s', equal_weighting)

        if self.is_training:
            self.reader = reader.DataReader(self.data_dir, self.batch_size,
                                            self.img_height, self.img_width,
                                            self.seq_length, NUM_SCALES,
                                            self.file_extension,
                                            self.random_scale_crop,
                                            self.flipping_mode,
                                            self.random_color,
                                            self.imagenet_norm,
                                            self.shuffle,
                                            self.input_file,
                                            self.seg_align_type)
            self.build_train_graph()
        else:
            self.build_depth_test_graph()
            self.build_egomotion_test_graph()

        # At this point, the model is ready. Print some info on model params.
        util.count_parameters()

    def build_train_graph(self):
        self.build_inference_for_training()
        self.build_loss()
        self.build_train_op()
        if self.build_sum:
            self.build_summaries()

    def build_inference_for_training(self):
        """Invokes depth and ego-motion networks."""
        if self.is_training:
            (self.image_stack, self.image_stack_norm, self.seg_stack,
             self.intrinsic_mat, self.intrinsic_mat_inv) = self.reader.read_data()

        with tf.variable_scope('depth_prediction'):
            # Organized by ...[i][scale].  Note that the order is flipped in
            # variables in build_loss() below.
            self.disp = {}
            self.depth = {}
            self.depth_upsampled = {}
            self.object_depth_loss = 0.0
            # Organized by [i].
            disp_bottlenecks = [None] * self.seq_length

            for i in range(self.seq_length):
                image = self.image_stack_norm[:, :, :, 3 * i:3 * (i + 1)]

                multiscale_disps_i, disp_bottlenecks[i] = nets.disp_net(
                    self.architecture, image, self.use_skip,
                    self.weight_reg, True)

                multiscale_depths_i = [1.0 / d for d in multiscale_disps_i]
                self.disp[i] = multiscale_disps_i
                self.depth[i] = multiscale_depths_i
                if self.depth_upsampling:
                    self.depth_upsampled[i] = []
                    # Upsample low-resolution depth maps using differentiable bilinear
                    # interpolation.
                    for s in range(len(multiscale_depths_i)):
                        self.depth_upsampled[i].append(tf.image.resize_bilinear(
                            multiscale_depths_i[s], [self.img_height, self.img_width],
                            align_corners=True))

                # Reuse the same depth graph for all images.
                tf.get_variable_scope().reuse_variables()

        if self.handle_motion:
            # Define egomotion network. This network can see the whole scene except
            # for any moving objects as indicated by the provided segmentation masks.
            # To avoid the network getting clues of motion by tracking those masks, we
            # define the segmentation masks as the union temporally.
            with tf.variable_scope('egomotion_prediction'):
                base_input = self.image_stack_norm  # (B, H, W, 9)
                seg_input = self.seg_stack  # (B, H, W, 9)
                ref_zero = tf.constant(0, dtype=tf.uint8)
                # Motion model is currently defined for three-frame sequences.
                object_mask1 = tf.equal(seg_input[:, :, :, 0], ref_zero)
                object_mask2 = tf.equal(seg_input[:, :, :, 3], ref_zero)
                object_mask3 = tf.equal(seg_input[:, :, :, 6], ref_zero)
                mask_complete = tf.expand_dims(tf.logical_and(  # (B, H, W, 1)
                    tf.logical_and(object_mask1, object_mask2), object_mask3), axis=3)
                mask_complete = tf.tile(mask_complete, (1, 1, 1, 9))  # (B, H, W, 9)
                # Now mask out base_input.
                self.mask_complete = tf.to_float(mask_complete)
                self.base_input_masked = base_input * self.mask_complete  # [B, H, W, 9]

                self.egomotion = nets.egomotion_net(
                    image_stack=self.base_input_masked,
                    disp_bottleneck_stack=None,
                    joint_encoder=False,
                    seq_length=self.seq_length,
                    weight_reg=self.weight_reg,
                    same_trans_rot_scaling=self.same_trans_rot_scaling)

            # Define object motion network for refinement. This network only sees
            # one object at a time over the whole sequence, and tries to estimate its
            # motion. The sequence of images are the respective warped frames.

            # For each scale, contains batch_size elements of shape (N, 2, 6).
            self.object_transforms = {}
            # For each scale, contains batch_size elements of shape (N, H, W, 9).
            self.object_masks = {}
            self.object_masks_warped = {}
            # For each scale, contains batch_size elements of size N.
            self.object_ids = {}

            self.egomotions_seq = {}
            self.warped_seq = {}
            # For each scale, contains 3 elements of shape [B, H, W, 2]
            self.rigid_flow_seq = {}
            self.inputs_region_deformer_net = {}
            with tf.variable_scope('objectmotion_prediction'):
                # First, warp raw images according to overall egomotion.
                for s in range(NUM_SCALES):
                    self.warped_seq[s] = []
                    self.rigid_flow_seq[s] = []
                    self.egomotions_seq[s] = []
                    for source_index in range(self.seq_length):
                        egomotion_mat_i_1 = project.get_transform_mat(
                            self.egomotion, source_index, 1, use_axis_angle=self.use_axis_angle)

                        # The gradient of egomotion network should only comes from background,
                        # stop gradient from objects
                        if self.stop_egomotion_gradient:
                            current_seg = self.seg_stack[:, :, :, source_index * 3]  # [B, H, W]
                            background_mask = tf.equal(current_seg,
                                                       tf.constant(0, dtype=tf.uint8))  # [B, H, W]
                            background_mask = tf.tile(tf.expand_dims(background_mask, axis=3),
                                                      (1, 1, 1, 3))  # [B, H, W, 3]
                            background_mask = tf.to_float(background_mask)

                            background_mask_warped, _ = (
                                project.inverse_warp(
                                    background_mask,
                                    self.depth_upsampled[1][s],
                                    egomotion_mat_i_1,
                                    self.intrinsic_mat[:, 0, :, :],
                                    self.intrinsic_mat_inv[:, 0, :, :]))
                            # Stop gradient for mask
                            background_mask_warped = tf.stop_gradient(background_mask_warped)

                            background_warped, _ = (
                                project.inverse_warp(
                                    self.image_stack[:, :, :, source_index * 3:(source_index + 1) * 3],
                                    self.depth_upsampled[1][s],
                                    egomotion_mat_i_1,
                                    self.intrinsic_mat[:, 0, :, :],
                                    self.intrinsic_mat_inv[:, 0, :, :]))

                            obj_warped, _ = (
                                project.inverse_warp(
                                    self.image_stack[:, :, :, source_index * 3:(source_index + 1) * 3],
                                    self.depth_upsampled[1][s],
                                    tf.stop_gradient(egomotion_mat_i_1),  # stop gradient from objects
                                    self.intrinsic_mat[:, 0, :, :],
                                    self.intrinsic_mat_inv[:, 0, :, :]))

                            warped_image_i_1 = background_warped * background_mask_warped + \
                                               obj_warped * (1.0 - background_mask_warped)

                            background_rigid_flow = project.compute_rigid_flow(
                                self.depth_upsampled[1][s],
                                egomotion_mat_i_1,
                                self.intrinsic_mat[:, 0, :, :],
                                self.intrinsic_mat_inv[:, 0, :, :]
                            )  # [B, H, W, 2]

                            obj_rigid_flow = project.compute_rigid_flow(
                                self.depth_upsampled[1][s],
                                tf.stop_gradient(egomotion_mat_i_1),  # stop gradients for objects
                                self.intrinsic_mat[:, 0, :, :],
                                self.intrinsic_mat_inv[:, 0, :, :]
                            )

                            rigid_flow_i_1 = background_rigid_flow * background_mask[:, :, :, :2] + \
                                             obj_rigid_flow * (1.0 - background_mask[:, :, :, :2])
                        else:
                            warped_image_i_1, _ = (
                                project.inverse_warp(
                                    self.image_stack[:, :, :, source_index * 3:(source_index + 1) * 3],
                                    self.depth_upsampled[1][s],
                                    egomotion_mat_i_1,
                                    self.intrinsic_mat[:, 0, :, :],
                                    self.intrinsic_mat_inv[:, 0, :, :]))

                            rigid_flow_i_1 = project.compute_rigid_flow(
                                self.depth_upsampled[1][s],
                                egomotion_mat_i_1,
                                self.intrinsic_mat[:, 0, :, :],
                                self.intrinsic_mat_inv[:, 0, :, :])

                        self.warped_seq[s].append(warped_image_i_1)
                        self.rigid_flow_seq[s].append(rigid_flow_i_1)
                        self.egomotions_seq[s].append(egomotion_mat_i_1)

                    # Second, for every object in the segmentation mask, take its mask and
                    # warp it according to the egomotion estimate. Then put a threshold to
                    # binarize the warped result. Use this mask to mask out background and
                    # other objects, and pass the filtered image to the region deformer
                    # network.
                    self.object_transforms[s] = []
                    self.object_masks[s] = []
                    self.object_ids[s] = []
                    self.object_masks_warped[s] = []
                    self.inputs_region_deformer_net[s] = {}

                    for i in range(self.batch_size):
                        seg_sequence = self.seg_stack[i]  # (H, W, 9=3*3)
                        # Backgound is 0, include 0 here
                        object_ids = tf.unique(tf.reshape(seg_sequence, [-1]))[0]

                        self.object_ids[s].append(object_ids)
                        color_stack = []
                        mask_stack = []
                        mask_stack_warped = []
                        for j in range(self.seq_length):
                            current_image = self.warped_seq[s][j][i]  # (H, W, 3)
                            current_seg = seg_sequence[:, :, j * 3:(j + 1) * 3]  # (H, W, 3)

                            # When enforcing object depth prior, exclude objects when computing
                            # neighboring mask
                            background = tf.equal(current_seg[:, :, 0],
                                                  tf.constant(0, dtype=tf.uint8))  # [H, W]

                            def process_obj_mask_warp(obj_id):
                                """Performs warping of the individual object masks."""
                                obj_mask = tf.to_float(tf.equal(current_seg, obj_id))
                                # Warp obj_mask according to overall egomotion.
                                obj_mask_warped, _ = (
                                    project.inverse_warp(
                                        tf.expand_dims(obj_mask, axis=0),
                                        # Middle frame, highest scale, batch element i:
                                        tf.expand_dims(self.depth_upsampled[1][s][i], axis=0),
                                        # Matrix for warping j into middle frame, batch elem. i:
                                        tf.expand_dims(self.egomotions_seq[s][j][i], axis=0),
                                        tf.expand_dims(self.intrinsic_mat[i, 0, :, :], axis=0),
                                        tf.expand_dims(self.intrinsic_mat_inv[i, 0, :, :],
                                                       axis=0)))

                                obj_mask_warped = tf.squeeze(obj_mask_warped, axis=0)  # specify axis=0
                                obj_mask_binarized = tf.greater(  # Threshold to binarize mask.
                                    obj_mask_warped, tf.constant(0.5))
                                return tf.to_float(obj_mask_binarized)  # [H, W, 3]

                            def process_obj_mask(obj_id):
                                """Returns the individual object masks separately."""
                                return tf.to_float(tf.equal(current_seg, obj_id))

                            object_masks = tf.map_fn(  # (N, H, W, 3)
                                process_obj_mask, object_ids, dtype=tf.float32)

                            if self.object_depth_weight > 0:
                                # The inverse depth of a moving object should be larger or equal to
                                # its horizontal surrounding environment
                                depth_pred = self.depth_upsampled[j][s][i]  # [H, W, 1]

                                def get_obj_losses(obj_mask):
                                    # Note obj_mask includes background

                                    # Find width of segment
                                    coords = tf.where(tf.greater(
                                        obj_mask[:, :, 0], tf.constant(0.5, dtype=tf.float32)
                                    ))  # [num_true, 2]
                                    y_max = tf.to_int32(tf.reduce_max(coords[:, 0]))
                                    y_min = tf.to_int32(tf.reduce_min(coords[:, 0]))
                                    x_max = tf.to_int32(tf.reduce_max(coords[:, 1]))
                                    x_min = tf.to_int32(tf.reduce_min(coords[:, 1]))

                                    neighbor_pixel = 10  # empirical value

                                    id_x_min = tf.maximum(0, x_min - neighbor_pixel)
                                    id_x_max = tf.minimum(self.img_width - 1, x_max + neighbor_pixel)

                                    slice1 = tf.zeros([y_min, self.img_width])

                                    slice2_1 = tf.zeros([y_max - y_min + 1, id_x_min])
                                    slice2_2 = tf.ones([y_max - y_min + 1,
                                                        (id_x_max - id_x_min + 1)])  # neighbor
                                    slice2_3 = tf.zeros([y_max - y_min + 1,
                                                         self.img_width - 1 - id_x_max])
                                    slice2 = tf.concat([slice2_1, slice2_2, slice2_3],
                                                       axis=1)  # [y_max - y_min, W]
                                    slice3 = tf.zeros([self.img_height - 1 - y_max, self.img_width])
                                    neighbor_mask = tf.concat([slice1, slice2, slice3],
                                                              axis=0)  # [H, W]

                                    neighbor_mask = neighbor_mask * (tf.to_float(
                                        tf.less(obj_mask[:, :, 0], tf.constant(0.5, dtype=tf.float32))
                                    ))

                                    # Handle overlapping objects
                                    if self.exclude_object_mask:
                                        neighbor_mask = neighbor_mask * tf.to_float(background)  # [H, W]

                                    neighbor_depth = tf.boolean_mask(
                                        depth_pred,
                                        tf.greater(
                                            tf.reshape(neighbor_mask,
                                                       (self.img_height, self.img_width, 1)),
                                            tf.constant(0.5, dtype=tf.float32)))
                                    reference_depth = tf.boolean_mask(
                                        depth_pred, tf.greater(
                                            tf.reshape(obj_mask[:, :, 0],
                                                       (self.img_height, self.img_width, 1)),
                                            tf.constant(0.5, dtype=tf.float32)))

                                    neighbor_mean = tf.reduce_mean(neighbor_depth)
                                    reference_mean = tf.reduce_mean(reference_depth)

                                    # Soft constraint
                                    loss = tf.maximum(reference_mean - neighbor_mean - self.object_depth_threshold,
                                                      tf.constant(0.0, dtype=tf.float32))
                                    return loss

                                losses = tf.map_fn(get_obj_losses, object_masks, dtype=tf.float32)
                                # Remove background, whose id is 0
                                self.object_depth_loss += tf.reduce_mean(tf.sign(tf.to_float(object_ids)) * losses)

                            object_masks_warped = tf.map_fn(  # (N, H, W, 3)
                                process_obj_mask_warp, object_ids, dtype=tf.float32)

                            # When warping object mask, stop gradient of depth and egomotion
                            if self.stop_egomotion_gradient:
                                object_masks_warped = tf.stop_gradient(object_masks_warped)

                            filtered_images = tf.map_fn(
                                lambda mask: current_image * mask, object_masks_warped,
                                dtype=tf.float32)  # (N, H, W, 3)
                            color_stack.append(filtered_images)
                            mask_stack.append(object_masks)
                            mask_stack_warped.append(object_masks_warped)

                        # For this batch-element, if there are N moving objects,
                        # color_stack, mask_stack and mask_stack_warped contain both
                        # seq_length elements of shape (N, H, W, 3).
                        # We can now concatenate them on the last axis, creating a tensor of
                        # (N, H, W, 3*3 = 9), and, assuming N does not get too large so that
                        # we have enough memory, pass them in a single batch to the region
                        # deformer network.
                        mask_stack = tf.concat(mask_stack, axis=3)  # (N, H, W, 9)
                        mask_stack_warped = tf.concat(mask_stack_warped, axis=3)
                        color_stack = tf.concat(color_stack, axis=3)  # (N, H, W, 9)

                        if self.stop_egomotion_gradient:
                            # Gradient has been stopped before
                            image_stack = color_stack
                        else:
                            image_stack = tf.stop_gradient(color_stack)

                        all_transforms = nets.region_deformer_net(
                            image_stack=image_stack,
                            disp_bottleneck_stack=None,
                            joint_encoder=False,  # joint encoder not supported.
                            seq_length=self.seq_length,
                            weight_reg=self.weight_reg,
                            trans_params_size=self.trans_params_size,
                            region_deformer_scaling=self.region_deformer_scaling)
                        # all_transforms of shape (N, 2, 32)

                        self.object_transforms[s].append(all_transforms)
                        self.object_masks[s].append(mask_stack)
                        self.object_masks_warped[s].append(mask_stack_warped)
                        self.inputs_region_deformer_net[s][i] = color_stack
                        tf.get_variable_scope().reuse_variables()
        else:
            # Don't handle motion, classic model formulation.
            with tf.name_scope('egomotion_prediction'):
                if self.joint_encoder:
                    # Re-arrange disp_bottleneck_stack to be of shape
                    # [B, h_hid, w_hid, c_hid * seq_length]. Currently, it is a list with
                    # seq_length elements, each of dimension [B, h_hid, w_hid, c_hid].
                    disp_bottleneck_stack = tf.concat(disp_bottlenecks, axis=3)
                else:
                    disp_bottleneck_stack = None
                self.egomotion = nets.egomotion_net(
                    image_stack=self.image_stack_norm,
                    disp_bottleneck_stack=disp_bottleneck_stack,
                    joint_encoder=self.joint_encoder,
                    seq_length=self.seq_length,
                    weight_reg=self.weight_reg,
                    same_trans_rot_scaling=self.same_trans_rot_scaling)

    def build_loss(self):
        """Adds ops for computing loss."""
        with tf.name_scope('compute_loss'):
            self.reconstr_loss = 0
            self.smooth_loss = 0
            self.ssim_loss = 0

            # self.images is organized by ...[scale][B, h, w, seq_len * 3].
            self.images = [None for _ in range(NUM_SCALES)]
            # Following nested lists are organized by ...[scale][source-target].
            self.warped_image = [{} for _ in range(NUM_SCALES)]
            self.bg_warped_image = [{} for _ in range(NUM_SCALES)]  # background warped images
            self.warp_mask = [{} for _ in range(NUM_SCALES)]
            self.warp_error = [{} for _ in range(NUM_SCALES)]
            self.ssim_error = [{} for _ in range(NUM_SCALES)]

            self.middle_frame_index = util.get_seq_middle(self.seq_length)

            # Compute losses at each scale.
            for s in range(NUM_SCALES):
                # Scale image stack.
                if s == 0:  # Just as a precaution. TF often has interpolation bugs.
                    self.images[s] = self.image_stack
                else:
                    height_s = int(self.img_height / (2 ** s))
                    width_s = int(self.img_width / (2 ** s))
                    self.images[s] = tf.image.resize_bilinear(
                        self.image_stack, [height_s, width_s], align_corners=True)

                # Smoothness.
                if self.smooth_weight > 0:
                    for i in range(self.seq_length):
                        # When computing minimum loss, use the depth map from the middle
                        # frame only.
                        if not self.compute_minimum_loss or i == self.middle_frame_index:
                            disp_smoothing = self.disp[i][s]
                            if self.depth_normalization:
                                # Perform depth normalization, dividing by the mean.
                                mean_disp = tf.reduce_mean(disp_smoothing, axis=[1, 2, 3],
                                                           keep_dims=True)
                                disp_input = disp_smoothing / mean_disp
                            else:
                                disp_input = disp_smoothing
                            scaling_f = (1.0 if self.equal_weighting else 1.0 / (2 ** s))
                            self.smooth_loss += scaling_f * self.depth_smoothness(
                                disp_input, self.images[s][:, :, :, 3 * i:3 * (i + 1)])

                # Image reconstruction loss
                # i is source, j is target
                for i in range(self.seq_length):
                    for j in range(self.seq_length):
                        if i == j:
                            continue

                        # When computing minimum loss, only consider the middle frame as
                        # target.
                        if self.compute_minimum_loss and j != self.middle_frame_index:
                            continue
                        # We only consider adjacent frames, unless either
                        # compute_minimum_loss is on (where the middle frame is matched with
                        # all other frames) or exhaustive_mode is on (where all frames are
                        # matched with each other).
                        if (not self.compute_minimum_loss and not self.exhaustive_mode and
                                abs(i - j) != 1):
                            continue

                        selected_scale = 0 if self.depth_upsampling else s
                        source = self.images[selected_scale][:, :, :, 3 * i:3 * (i + 1)]
                        target = self.images[selected_scale][:, :, :, 3 * j:3 * (j + 1)]

                        if self.depth_upsampling:
                            target_depth = self.depth_upsampled[j][s]
                        else:
                            target_depth = self.depth[j][s]

                        key = '%d-%d' % (i, j)

                        if self.handle_motion:
                            # self.seg_stack of shape (B, H, W, 9).
                            # target_depth corresponds to middle frame, of shape (B, H, W, 1).

                            # Now incorporate the other warping results, performed according
                            # to the object motion network's predictions.
                            # self.object_masks batch_size elements of (N, H, W, 9).
                            # self.object_masks_warped batch_size elements of (N, H, W, 9).
                            # self.object_transforms batch_size elements of (N, 2, 6).
                            self.all_batches = []
                            self.residual_flow_all_batches = []
                            self.full_flow_all_batches = []
                            self.bg_warped_all_batches = []
                            for batch_s in range(self.batch_size):
                                # To warp i into j, first take the base warping (this is the
                                # full image i warped into j using only the egomotion estimate).
                                base_warping = self.warped_seq[s][i][batch_s]  # [H, W, 3]
                                base_rigid_flow = self.rigid_flow_seq[s][i][batch_s]  # [H, W, 2]

                                transform_matrices_thisbatch = tf.map_fn(
                                    lambda transform: project.get_region_deformer_params(
                                        tf.expand_dims(transform, axis=0), i, j)[0],
                                    self.object_transforms[0][batch_s])  # [N, 16, 2]

                                if self.use_rigid_residual_flow:
                                    def compute_residual_flow(trans_mat):
                                        residual_flow = project.compute_residual_flow(
                                            tf.expand_dims(base_warping, axis=0),
                                            tf.expand_dims(trans_mat, axis=0))  # [1, H, W, 2]
                                        return residual_flow

                                    def inverse_warp_wrapper(trans_mat):
                                        """Warp by rigid plus residual flow"""
                                        residual_flow = project.compute_residual_flow(
                                            tf.expand_dims(base_warping, axis=0),
                                            tf.expand_dims(trans_mat, axis=0))  # [1, H, W, 2]
                                        warped_image, _ = project.rigid_residual_flow_warp(
                                            tf.expand_dims(source[batch_s], axis=0),
                                            tf.expand_dims(base_rigid_flow, axis=0),
                                            residual_flow)
                                        return warped_image
                                else:
                                    def inverse_warp_wrapper(trans_mat):
                                        """Region Deformer"""
                                        transformed_image, _ = project.region_deformer(
                                            tf.expand_dims(base_warping, axis=0),
                                            tf.expand_dims(trans_mat, axis=0),
                                            residual=self.residual_deformer)
                                        return transformed_image

                                if self.use_rigid_residual_flow:
                                    residual_flow_thisbatch = tf.map_fn(
                                        compute_residual_flow, transform_matrices_thisbatch,
                                        dtype=tf.float32)
                                    residual_flow_thisbatch = residual_flow_thisbatch[:, 0, :, :, :]  # [N, H, W, 2]

                                warped_images_thisbatch = tf.map_fn(
                                    inverse_warp_wrapper, transform_matrices_thisbatch,
                                    dtype=tf.float32)
                                warped_images_thisbatch = warped_images_thisbatch[:, 0, :, :, :]
                                # warped_images_thisbatch is now of shape (N, H, W, 3).

                                # Combine warped frames into a single one, using the object
                                # masks. Result should be (1, 128, 416, 3).
                                # Essentially, we here want to sum them all up, filtered by the
                                # respective object masks.
                                mask_base_valid_source = tf.equal(
                                    self.seg_stack[batch_s, :, :, i * 3:(i + 1) * 3],
                                    tf.constant(0, dtype=tf.uint8))
                                mask_base_valid_target = tf.equal(
                                    self.seg_stack[batch_s, :, :, j * 3:(j + 1) * 3],
                                    tf.constant(0, dtype=tf.uint8))
                                mask_valid = tf.logical_and(
                                    mask_base_valid_source, mask_base_valid_target)
                                self.base_warping = base_warping * tf.to_float(mask_valid)
                                background = tf.expand_dims(self.base_warping, axis=0)  # [1, H, W, 3]

                                def construct_const_filter_tensor(obj_id):
                                    return tf.fill(
                                        dims=[self.img_height, self.img_width, 3],
                                        value=tf.sign(obj_id)) * tf.to_float(
                                        tf.equal(self.seg_stack[batch_s, :, :, 3:6],
                                                 tf.cast(obj_id, dtype=tf.uint8)))

                                filter_tensor = tf.map_fn(
                                    construct_const_filter_tensor,
                                    tf.to_float(self.object_ids[s][batch_s]))
                                filter_tensor = tf.stack(filter_tensor, axis=0)  # [N, H, W, 3]

                                if self.use_rigid_residual_flow:
                                    residual_flow_to_add = tf.reduce_sum(
                                        tf.multiply(residual_flow_thisbatch, filter_tensor[:, :, :, :2]),
                                        axis=0, keep_dims=True)  # [1, H, W, 2]

                                    # Warp by rigid flow plus residual flow
                                    combined, _ = project.rigid_residual_flow_warp(
                                        tf.expand_dims(source[batch_s], axis=0),
                                        tf.expand_dims(base_rigid_flow, axis=0),
                                        residual_flow_to_add)
                                else:
                                    objects_to_add = tf.reduce_sum(
                                        tf.multiply(warped_images_thisbatch, filter_tensor),
                                        axis=0, keep_dims=True)  # [1, H, W, 3]
                                    combined = background + objects_to_add

                                self.all_batches.append(combined)
                                self.bg_warped_all_batches.append(background)

                            # Now of shape (B, 128, 416, 3).
                            self.warped_image[s][key] = tf.concat(self.all_batches, axis=0)
                            self.bg_warped_image[s][key] = tf.concat(self.bg_warped_all_batches, axis=0)

                        else:
                            # Don't handle motion, classic model formulation.
                            egomotion_mat_i_j = project.get_transform_mat(
                                self.egomotion, i, j, use_axis_angle=self.use_axis_angle)
                            # Inverse warp the source image to the target image frame for
                            # photometric consistency loss.
                            self.warped_image[s][key], self.warp_mask[s][key] = (
                                project.inverse_warp(
                                    source,
                                    target_depth,
                                    egomotion_mat_i_j,
                                    self.intrinsic_mat[:, selected_scale, :, :],
                                    self.intrinsic_mat_inv[:, selected_scale, :, :]))

                        # Reconstruction loss.
                        self.warp_error[s][key] = tf.abs(self.warped_image[s][key] - target)
                        if not self.compute_minimum_loss:
                            self.reconstr_loss += tf.reduce_mean(
                                self.warp_error[s][key] * self.warp_mask[s][key])
                        # SSIM.
                        if self.ssim_weight > 0:
                            self.ssim_error[s][key] = self.ssim(self.warped_image[s][key],
                                                                target)
                            if not self.compute_minimum_loss:
                                ssim_mask = slim.avg_pool2d(self.warp_mask[s][key], 3, 1,
                                                            'VALID')
                                self.ssim_loss += tf.reduce_mean(
                                    self.ssim_error[s][key] * ssim_mask)

                # If the minimum loss should be computed, the loss calculation has been
                # postponed until here.
                if self.compute_minimum_loss:
                    for frame_index in range(self.middle_frame_index):
                        key1 = '%d-%d' % (frame_index, self.middle_frame_index)
                        key2 = '%d-%d' % (self.seq_length - frame_index - 1,
                                          self.middle_frame_index)
                        logging.info('computing min error between %s and %s', key1, key2)
                        min_error = tf.minimum(self.warp_error[s][key1],
                                               self.warp_error[s][key2])
                        self.reconstr_loss += tf.reduce_mean(min_error)
                        if self.ssim_weight > 0:  # Also compute the minimum SSIM loss.
                            min_error_ssim = tf.minimum(self.ssim_error[s][key1],
                                                        self.ssim_error[s][key2])
                            self.ssim_loss += tf.reduce_mean(min_error_ssim)

            # Build the total loss as composed of L1 reconstruction, SSIM, smoothing
            # and object prior loss as appropriate.
            self.reconstr_loss *= self.reconstr_weight
            self.total_loss = self.reconstr_loss
            if self.smooth_weight > 0:
                self.smooth_loss *= self.smooth_weight
                self.total_loss += self.smooth_loss
            if self.ssim_weight > 0:
                self.ssim_loss *= self.ssim_weight
                self.total_loss += self.ssim_loss
            if self.object_depth_weight > 0:
                self.object_depth_loss *= self.object_depth_weight
                self.total_loss += self.object_depth_loss

    def gradient_x(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def gradient_y(self, img):
        return img[:, :-1, :, :] - img[:, 1:, :, :]

    def depth_smoothness(self, depth, img):
        """Computes image-aware depth smoothness loss."""
        depth_dx = self.gradient_x(depth)
        depth_dy = self.gradient_y(depth)
        image_dx = self.gradient_x(img)
        image_dy = self.gradient_y(img)
        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keep_dims=True))
        smoothness_x = depth_dx * weights_x
        smoothness_y = depth_dy * weights_y
        return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    def ssim(self, x, y):
        """Computes a differentiable structured image similarity measure."""
        c1 = 0.01 ** 2  # As defined in SSIM to stabilize div. by small denominator.
        c2 = 0.03 ** 2
        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return tf.clip_by_value((1 - ssim) / 2, 0, 1)

    def build_train_op(self):
        with tf.name_scope('train_op'):
            optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            self.train_op = slim.learning.create_train_op(self.total_loss, optim)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(
                self.global_step, self.global_step + 1)

    def build_summaries(self):
        """Adds scalar and image summaries for TensorBoard."""
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('reconstr_loss', self.reconstr_loss)
        if self.smooth_weight > 0:
            tf.summary.scalar('smooth_loss', self.smooth_loss)
        if self.ssim_weight > 0:
            tf.summary.scalar('ssim_loss', self.ssim_loss)

        if self.object_depth_weight > 0:
            tf.summary.scalar('object_depth_loss', self.object_depth_loss)

        for s in range(NUM_SCALES):
            for i in range(self.seq_length):
                tf.summary.image('scale%d_image%d' % (s, i),
                                 self.images[s][:, :, :, 3 * i:3 * (i + 1)])
                if i in self.depth:
                    tf.summary.histogram('scale%d_depth%d' % (s, i), self.depth[i][s])
                    tf.summary.histogram('scale%d_disp%d' % (s, i), self.disp[i][s])
                    tf.summary.image('scale%d_disparity%d' % (s, i), self.disp[i][s])

            for key in self.warped_image[s]:
                tf.summary.image('scale%d_warped_image%s' % (s, key),
                                 self.warped_image[s][key])
                tf.summary.image('scale%d_warp_error%s' % (s, key),
                                 self.warp_error[s][key])
                if self.ssim_weight > 0:
                    tf.summary.image('scale%d_ssim_error%s' % (s, key),
                                     self.ssim_error[s][key])

    def build_depth_test_graph(self):
        """Builds depth model reading from placeholders."""
        with tf.variable_scope('depth_prediction'):
            input_image = tf.placeholder(
                tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
                name='raw_input')
            if self.imagenet_norm:
                input_image = (input_image - reader.IMAGENET_MEAN) / reader.IMAGENET_SD
            est_disp, _ = nets.disp_net(architecture=self.architecture,
                                        image=input_image,
                                        use_skip=self.use_skip,
                                        weight_reg=self.weight_reg,
                                        is_training=True)
        est_depth = 1.0 / est_disp[0]
        self.input_image = input_image
        self.est_depth = est_depth

    def build_single_depth_test_graph(self):
        """Assume batch size is 1"""
        with tf.variable_scope('depth_prediction'):
            input_image = tf.placeholder(
                tf.float32, [1, self.img_height, self.img_width, 3],
                name='raw_input')
            if self.imagenet_norm:
                input_image = (input_image - reader.IMAGENET_MEAN) / reader.IMAGENET_SD

            tf.get_variable_scope().reuse_variables()  # Note: reuse variable
            est_disp, _ = nets.disp_net(architecture=self.architecture,
                                        image=input_image,
                                        use_skip=self.use_skip,
                                        weight_reg=self.weight_reg,
                                        is_training=True)

        est_depth = 1.0 / est_disp[0]
        self.input_image = input_image
        self.est_depth = est_depth

    def build_egomotion_test_graph(self):
        """Builds egomotion model reading from placeholders."""
        input_image_stack = tf.placeholder(
            tf.float32,
            [1, self.img_height, self.img_width, self.seq_length * 3],
            name='raw_input')
        input_bottleneck_stack = None

        if self.imagenet_norm:
            im_mean = tf.tile(
                tf.constant(reader.IMAGENET_MEAN), multiples=[self.seq_length])
            im_sd = tf.tile(
                tf.constant(reader.IMAGENET_SD), multiples=[self.seq_length])
            input_image_stack = (input_image_stack - im_mean) / im_sd

        if self.joint_encoder:
            # Pre-compute embeddings here.
            with tf.variable_scope('depth_prediction', reuse=True):
                input_bottleneck_stack = []
                encoder_selected = nets.encoder(self.architecture)
                for i in range(self.seq_length):
                    input_image = input_image_stack[:, :, :, i * 3:(i + 1) * 3]
                    tf.get_variable_scope().reuse_variables()
                    embedding, _ = encoder_selected(
                        target_image=input_image,
                        weight_reg=self.weight_reg,
                        is_training=True)
                    input_bottleneck_stack.append(embedding)
                input_bottleneck_stack = tf.concat(input_bottleneck_stack, axis=3)

        with tf.variable_scope('egomotion_prediction'):
            est_egomotion = nets.egomotion_net(
                image_stack=input_image_stack,
                disp_bottleneck_stack=input_bottleneck_stack,
                joint_encoder=self.joint_encoder,
                seq_length=self.seq_length,
                weight_reg=self.weight_reg,
                same_trans_rot_scaling=self.same_trans_rot_scaling)
        self.input_image_stack = input_image_stack
        self.est_egomotion = est_egomotion

    def inference_depth(self, inputs, sess):
        return sess.run(self.est_depth, feed_dict={self.input_image: inputs})

    def inference_single_depth(self, inputs, sess):
        return sess.run(self.est_depth, feed_dict={self.input_image: inputs})

    def inference_egomotion(self, inputs, sess):
        return sess.run(
            self.est_egomotion, feed_dict={self.input_image_stack: inputs})
