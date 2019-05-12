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

"""Common utilities for data pre-processing, e.g. matching moving object across frames.
Based on struct2depth"""

import numpy as np
from glob import glob
from PIL import Image


def compute_mask_bbox_height(mask):
    """Compute mask's bounding box's height
    Ref: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    Args:
        mask: [H, W]
    """
    rows = np.any(mask, axis=1)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    height = rmax - rmin
    return height


def compute_overlap(mask1, mask2, height_threshold=20):
    """Empirically filter out small objects whose height less than 20 pixels"""
    height1 = compute_mask_bbox_height(mask1)
    height2 = compute_mask_bbox_height(mask2)
    # Remove small objects based height threshold
    if height1 < height_threshold or height2 < height_threshold:
        iou = 0
    else:
        # Use IoU here.
        iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
    return iou


def align(seg_img1, seg_img2, seg_img3, threshold_same=0.3):
    res_img1 = np.zeros_like(seg_img1)
    res_img2 = np.zeros_like(seg_img2)
    res_img3 = np.zeros_like(seg_img3)
    remaining_objects2 = list(np.unique(seg_img2.flatten()))
    remaining_objects3 = list(np.unique(seg_img3.flatten()))
    for seg_id in np.unique(seg_img1):
        # See if we can find correspondences to seg_id in seg_img2.
        max_overlap2 = float('-inf')
        max_segid2 = -1
        for seg_id2 in remaining_objects2:
            overlap = compute_overlap(seg_img1 == seg_id, seg_img2 == seg_id2)
            if overlap > max_overlap2:
                max_overlap2 = overlap
                max_segid2 = seg_id2
        if max_overlap2 > threshold_same:
            max_overlap3 = float('-inf')
            max_segid3 = -1
            for seg_id3 in remaining_objects3:
                overlap = compute_overlap(seg_img2 == max_segid2, seg_img3 == seg_id3)
                if overlap > max_overlap3:
                    max_overlap3 = overlap
                    max_segid3 = seg_id3
            if max_overlap3 > threshold_same:
                res_img1[seg_img1 == seg_id] = seg_id
                res_img2[seg_img2 == max_segid2] = seg_id
                res_img3[seg_img3 == max_segid3] = seg_id
                remaining_objects2.remove(max_segid2)
                remaining_objects3.remove(max_segid3)
    return res_img1, res_img2, res_img3


def align_segs(data_dir, seg_postfix='-raw-fseg.png', save_postfix='-h20-fseg.png'):
    """Align all segments across frames
    Args:
        data_dir: path to segments
        seg_postfix: postfix of segments filename before alignment
        save_postfix: postrfix of segments filename after alignment
    """
    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]

    # Find all segments
    seg_files = sorted(glob(data_dir + '/*/*' + seg_postfix))

    img_width = 416  # KITTI and Cityscapes datasets

    print('Segmented images: {}'.format(len(seg_files)))

    for i in range(len(seg_files)):
        if i % 100 == 0:
            print('processing {}/{}'.format(i, len(seg_files)))

        filename = seg_files[i]
        image = np.array(Image.open(filename))

        seg_img1 = image[:, :img_width]
        seg_img2 = image[:, img_width:img_width * 2]
        seg_img3 = image[:, img_width * 2:]

        # Align
        res_img1, res_img2, res_img3 = align(seg_img1, seg_img2, seg_img3)
        save_path = filename.replace(seg_postfix, save_postfix)

        img_stack = np.concatenate([res_img1, res_img2, res_img3], axis=1)
        Image.fromarray(img_stack).save(save_path)
