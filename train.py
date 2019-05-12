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

"""Train the model. Please refer to README for example usage.
Based on struct2depth"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import model
import nets
import reader
import util

import fnmatch

gfile = tf.gfile
MAX_TO_KEEP = 1000000  # Maximum number of checkpoints to keep.

flags.DEFINE_string('data_dir', None, 'Preprocessed data.')
flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')
flags.DEFINE_float('learning_rate', 0.0002, 'Adam learning rate.')
flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')
flags.DEFINE_float('ssim_weight', 0.15, 'SSIM loss weight.')
flags.DEFINE_float('smooth_weight', 0.04, 'Smoothness loss weight.')
flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_enum('architecture', nets.RESNET, nets.ARCHITECTURES,
                  'Defines the architecture to use for the depth prediction '
                  'network. Defaults to ResNet-based encoder and accompanying '
                  'decoder.')
flags.DEFINE_boolean('imagenet_norm', True, 'Whether to normalize the input '
                                            'images channel-wise so that they match the distribution '
                                            'most ImageNet-models were trained on.')
flags.DEFINE_float('weight_reg', 0.05, 'The amount of weight regularization to '
                                       'apply. This has no effect on the ResNet-based encoder '
                                       'architecture.')
flags.DEFINE_boolean('exhaustive_mode', False, 'Whether to exhaustively warp '
                                               'from any frame to any other instead of just considering '
                                               'adjacent frames. Where necessary, multiple egomotion '
                                               'estimates will be applied. Does not have an effect if '
                                               'compute_minimum_loss is enabled.')
flags.DEFINE_boolean('random_scale_crop', False, 'Whether to apply random '
                                                 'image scaling and center cropping during training.')
flags.DEFINE_enum('flipping_mode', reader.FLIP_RANDOM,
                  [reader.FLIP_RANDOM, reader.FLIP_ALWAYS, reader.FLIP_NONE],
                  'Determines the image flipping mode: if random, performs '
                  'on-the-fly augmentation. Otherwise, flips the input images '
                  'always or never, respectively.')
flags.DEFINE_string('pretrained_ckpt', None, 'Path to checkpoint with '
                                             'pretrained weights.  Do not include .data* extension.')
flags.DEFINE_string('imagenet_ckpt', None, 'Initialize the weights according '
                                           'to an ImageNet-pretrained checkpoint. Requires '
                                           'architecture to be ResNet-18.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save model '
                                            'checkpoints.')
flags.DEFINE_integer('train_steps', 300000, 'Number of training steps.')
flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')
flags.DEFINE_bool('depth_upsampling', True, 'Whether to apply depth '
                                            'upsampling of lower-scale representations before warping to '
                                            'compute reconstruction loss on full-resolution image.')
flags.DEFINE_bool('depth_normalization', True, 'Whether to apply depth '
                                               'normalization, that is, normalizing inverse depth '
                                               'prediction maps by their mean to avoid degeneration towards '
                                               'small values.')
flags.DEFINE_bool('compute_minimum_loss', True, 'Whether to take the '
                                                'element-wise minimum of the reconstruction/SSIM error in '
                                                'order to avoid overly penalizing dis-occlusion effects.')
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                                    'encoder-decoder architecture.')
flags.DEFINE_bool('equal_weighting', False, 'Whether to use equal weighting '
                                            'of the smoothing loss term, regardless of resolution.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                                          'between the depth and egomotion networks by using a joint '
                                          'encoder architecture. The egomotion network is then '
                                          'operating only on the hidden representation provided by the '
                                          'joint encoder.')
flags.DEFINE_bool('handle_motion', False, 'Whether to try to handle motion by '
                                          'using the provided segmentation masks.')

flags.DEFINE_bool('same_trans_rot_scaling', True,
                  'Whether to use the same scaling constant for translation and rotation')
flags.DEFINE_bool('residual_deformer', True, 'Whether to learn residual transform')
flags.DEFINE_string('seg_align_type', 'h20', 'Type of alignment when matching segmented objects')
flags.DEFINE_bool('use_rigid_residual_flow', True, 'Directly warp by rigid flow plus residual flow')
flags.DEFINE_float('region_deformer_scaling', 1.0, 'Scaling factor for region deformer output')
flags.DEFINE_float('object_depth_weight', 0.0, 'The inverse depth of moving objects should be larger or '
                                               'equal to its surrounding environment')
flags.DEFINE_float('object_depth_threshold', 0.01, 'Soft constraint for object depth assumption')
flags.DEFINE_float('discontinuity_weight', 0.0, 'Depth discontinuity when encountering segment edge')
flags.DEFINE_string('input_file', 'train', 'Dataset split for training')
flags.DEFINE_bool('exclude_object_mask', True, 'When enforcing object depth prior, excluding object mask to'
                                               'handle objects overlapping')
flags.DEFINE_bool('stop_egomotion_gradient', True, 'The gradient of egomotion network only comes from '
                                                   'background, stop gradient of objects')
flags.DEFINE_bool('use_axis_angle', False, 'Use axis-angle to represent camera rotation')

flags.DEFINE_bool('build_summary', True, 'Whether to build summary when training, disable to save memory')
flags.DEFINE_string('master', 'local', 'Location of the session.')
flags.DEFINE_string('device', '0', 'GPU id')

FLAGS = flags.FLAGS
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('checkpoint_dir')


def main(_):
    # Fixed seed for repeatability
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    util.save_flags(FLAGS, FLAGS.checkpoint_dir)
    util.save_command(FLAGS.checkpoint_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

    if FLAGS.handle_motion and FLAGS.joint_encoder:
        raise ValueError('Using a joint encoder is currently not supported when '
                         'modeling object motion.')
    if FLAGS.handle_motion and FLAGS.seq_length != 3:
        raise ValueError('The current motion model implementation only supports '
                         'using a sequence length of three.')
    if FLAGS.handle_motion and not FLAGS.compute_minimum_loss:
        raise ValueError('Computing the minimum photometric loss is required when '
                         'enabling object motion handling.')
    if FLAGS.object_depth_weight > 0 and not FLAGS.handle_motion:
        raise ValueError('To enforce object depth prior, enable motion '
                         'handling.')
    if FLAGS.imagenet_ckpt and not FLAGS.imagenet_norm:
        logging.warning('When initializing with an ImageNet-pretrained model, it is '
                        'recommended to normalize the image inputs accordingly using '
                        'imagenet_norm.')
    if FLAGS.compute_minimum_loss and FLAGS.seq_length % 2 != 1:
        raise ValueError('Compute minimum loss requires using an odd number of '
                         'images in a sequence.')
    if FLAGS.architecture != nets.RESNET and FLAGS.imagenet_ckpt:
        raise ValueError('Can only load weights from pre-trained ImageNet model '
                         'when using ResNet-architecture.')
    if FLAGS.compute_minimum_loss and FLAGS.exhaustive_mode:
        raise ValueError('Exhaustive mode has no effect when compute_minimum_loss '
                         'is enabled.')
    if FLAGS.img_width % (2 ** 5) != 0 or FLAGS.img_height % (2 ** 5) != 0:
        logging.warning('Image size is not divisible by 2^5. For the architecture '
                        'employed, this could cause artefacts caused by resizing in '
                        'lower dimensions.')

    if not gfile.Exists(FLAGS.checkpoint_dir):
        gfile.MakeDirs(FLAGS.checkpoint_dir)

    train_model = model.Model(data_dir=FLAGS.data_dir,
                              file_extension=FLAGS.file_extension,
                              is_training=True,
                              input_file=FLAGS.input_file,
                              learning_rate=FLAGS.learning_rate,
                              beta1=FLAGS.beta1,
                              reconstr_weight=FLAGS.reconstr_weight,
                              smooth_weight=FLAGS.smooth_weight,
                              ssim_weight=FLAGS.ssim_weight,
                              object_depth_weight=FLAGS.object_depth_weight,
                              object_depth_threshold=FLAGS.object_depth_threshold,
                              exclude_object_mask=FLAGS.exclude_object_mask,
                              stop_egomotion_gradient=FLAGS.stop_egomotion_gradient,
                              batch_size=FLAGS.batch_size,
                              img_height=FLAGS.img_height,
                              img_width=FLAGS.img_width,
                              seq_length=FLAGS.seq_length,
                              architecture=FLAGS.architecture,
                              imagenet_norm=FLAGS.imagenet_norm,
                              weight_reg=FLAGS.weight_reg,
                              exhaustive_mode=FLAGS.exhaustive_mode,
                              random_scale_crop=FLAGS.random_scale_crop,
                              flipping_mode=FLAGS.flipping_mode,
                              depth_upsampling=FLAGS.depth_upsampling,
                              depth_normalization=FLAGS.depth_normalization,
                              compute_minimum_loss=FLAGS.compute_minimum_loss,
                              use_skip=FLAGS.use_skip,
                              use_axis_angle=FLAGS.use_axis_angle,
                              build_sum=FLAGS.build_summary,
                              joint_encoder=FLAGS.joint_encoder,
                              handle_motion=FLAGS.handle_motion,
                              equal_weighting=FLAGS.equal_weighting,
                              same_trans_rot_scaling=FLAGS.same_trans_rot_scaling,
                              residual_deformer=FLAGS.residual_deformer,
                              seg_align_type=FLAGS.seg_align_type,
                              use_rigid_residual_flow=FLAGS.use_rigid_residual_flow,
                              region_deformer_scaling=FLAGS.region_deformer_scaling)

    train(train_model, FLAGS.pretrained_ckpt, FLAGS.imagenet_ckpt,
          FLAGS.checkpoint_dir, FLAGS.train_steps, FLAGS.summary_freq)


def train(train_model, pretrained_ckpt, imagenet_ckpt, checkpoint_dir,
          train_steps, summary_freq):
    """Train model."""
    if checkpoint_dir.endswith('/'):
        checkpoint_dir = checkpoint_dir[:-1]  # remove '/'
    vars_to_restore = None
    if pretrained_ckpt is not None:
        vars_to_restore = util.get_vars_to_save_and_restore(pretrained_ckpt)
        ckpt_path = pretrained_ckpt
    elif imagenet_ckpt:
        vars_to_restore = util.get_imagenet_vars_to_restore(imagenet_ckpt)
        ckpt_path = imagenet_ckpt
    pretrain_restorer = tf.train.Saver(vars_to_restore)
    vars_to_save = util.get_vars_to_save_and_restore()
    vars_to_save[train_model.global_step.op.name] = train_model.global_step
    saver = tf.train.Saver(vars_to_save, max_to_keep=MAX_TO_KEEP)
    sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                             saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        if pretrained_ckpt is not None or imagenet_ckpt:
            logging.info('Restoring pretrained weights from %s', ckpt_path)
            pretrain_restorer.restore(sess, ckpt_path)

            # If the pretrained model's dir is the same as current checkpoint dir,
            # restore global_scale from pretrained checkpoint
            if os.path.dirname(ckpt_path) == checkpoint_dir:
                logging.info('Attempting to resume training from %s', ckpt_path)
                saver.restore(sess, ckpt_path)

        logging.info('Training...')
        start_time = time.time()
        last_summary_time = time.time()
        steps_per_epoch = train_model.reader.steps_per_epoch
        step = 1
        while step <= train_steps:
            fetches = {
                'train': train_model.train_op,
                'global_step': train_model.global_step,
                'incr_global_step': train_model.incr_global_step,
            }

            if step % summary_freq == 0:
                fetches['loss'] = train_model.total_loss
                fetches['summary'] = sv.summary_op

            results = sess.run(fetches)
            global_step = results['global_step']

            if step % summary_freq == 0:
                sv.summary_writer.add_summary(results['summary'], global_step)
                train_epoch = math.ceil(global_step / steps_per_epoch)
                train_step = global_step - (train_epoch - 1) * steps_per_epoch
                this_cycle = time.time() - last_summary_time
                last_summary_time += this_cycle
                logging.info(
                    'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
                    train_epoch, train_step, steps_per_epoch, this_cycle,
                    time.time() - start_time, results['loss'])

            # When dataset is too small, not save after every epoch
            if step % steps_per_epoch == 0:
                logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                           global_step=global_step)

            # Setting step to global_step allows for training for a total of
            # train_steps even if the program is restarted during training.
            step = global_step + 1


def collect_input_images(input_dir, input_list_file, file_extension):
    """Collects all input images that are to be processed."""
    if input_dir is not None:
        im_files = _recursive_glob(input_dir, '*.' + file_extension)
        basepath_in = os.path.normpath(input_dir)
    elif input_list_file is not None:
        im_files = util.read_text_lines(input_list_file)
        basepath_in = os.path.dirname(input_list_file)
        im_files = [os.path.join(basepath_in, f) for f in im_files]
    im_files = [f for f in im_files if 'disp' not in f and '-seg' not in f and
                '-fseg' not in f and '-flip' not in f]
    return sorted(im_files), basepath_in


def _recursive_glob(treeroot, pattern):
    results = []
    for base, _, files in os.walk(treeroot):
        files = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in files)
    return results


if __name__ == '__main__':
    app.run(main)
