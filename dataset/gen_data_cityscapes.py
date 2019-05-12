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

"""Offline data generation for the Cityscapes dataset.
Based on struct2depth to make fair comparison"""

import numpy as np
import cv2
import os
import glob

SKIP = 2  # to match KITTI frame rate


def crop(img, fx, fy, cx, cy, height, width):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1 - middle_perc
    half = left / 2
    a = img[int(img.shape[0] * half):int(img.shape[0] * (1 - half)), :]
    cy /= (1 / middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((float(height) * a.shape[1] / a.shape[0]))
    x_scaling = float(wdt) / a.shape[1]
    y_scaling = float(height) / a.shape[0]
    b = cv2.resize(a, (wdt, height))

    # Adjust intrinsics.
    fx *= x_scaling
    fy *= y_scaling
    cx *= x_scaling
    cy *= y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - width
    cx /= (b.shape[1] / width)
    c = b[:, int(remain / 2):b.shape[1] - int(remain / 2)]

    return c, fx, fy, cx, cy


def gen_data_cityscapes(input_dir, dump_root, height=128, width=416, seq_length=3,
                        sub_folder='train'):
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)

    dir_name = input_dir + '/leftImg8bit_sequence/' + sub_folder + '/*'
    print('Processing directory', dir_name)
    for location in glob.glob(input_dir + '/leftImg8bit_sequence/' + sub_folder + '/*'):
        location_name = os.path.basename(location)
        print('Processing location', location_name)
        files = sorted(glob.glob(location + '/*.png'))
        # Break down into sequences
        sequences = {}
        seq_nr = -1
        last_seq = ''
        last_imgnr = -1

        for i in range(len(files)):
            seq = os.path.basename(files[i]).split('_')[1]
            nr = int(os.path.basename(files[i]).split('_')[2])
            if seq != last_seq or last_imgnr + 1 != nr:
                seq_nr += 1
            last_imgnr = nr
            last_seq = seq
            if seq_nr not in sequences:
                sequences[seq_nr] = []
            sequences[seq_nr].append(files[i])

        for (k, v) in sequences.items():
            print('Processing sequence', k, 'with', len(v), 'elements...')
            output_dir = dump_root + '/' + location_name + '_' + str(k).zfill(3)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            files = sorted(v)
            triplet = []
            ct = 1

            # Find applicable intrinsics.
            for j in range(len(files)):
                osegname = os.path.basename(files[j]).split('_')[1]
                oimgnr = os.path.basename(files[j]).split('_')[2]
                applicable_intrinsics = input_dir + '/camera/' + sub_folder + '/' + location_name + '/' + \
                                        location_name + '_' + osegname + '_' + oimgnr + '_camera.json'
                # Get the intrinsics for one of the file of the sequence.
                if os.path.isfile(applicable_intrinsics):
                    f = open(applicable_intrinsics, 'r')
                    lines = f.readlines()
                    f.close()
                    lines = [line.rstrip() for line in lines]

                    fx = float(lines[11].split(': ')[1].replace(',', ''))
                    fy = float(lines[12].split(': ')[1].replace(',', ''))
                    cx = float(lines[13].split(': ')[1].replace(',', ''))
                    cy = float(lines[14].split(': ')[1].replace(',', ''))

            for j in range(0, len(files), SKIP):
                img = cv2.imread(files[j])

                smallimg, fx_this, fy_this, cx_this, cy_this = crop(img, fx, fy, cx, cy, height, width)
                triplet.append(smallimg)
                if len(triplet) == seq_length:
                    cmb = np.hstack(triplet)
                    cv2.imwrite(os.path.join(output_dir, str(ct).zfill(10) + '.png'), cmb)
                    f = open(os.path.join(output_dir, str(ct).zfill(10) + '_cam.txt'), 'w')
                    f.write(str(fx_this) + ',0.0,' + str(cx_this) + ',0.0,' + str(fy_this) + ',' + str(
                        cy_this) + ',0.0,0.0,1.0')
                    f.close()
                    del triplet[0]
                    ct += 1

    # Create file list for training. Be careful as it collects and includes all files recursively.
    fn = open(dump_root + '/' + sub_folder + '.txt', 'w')
    for f in glob.glob(dump_root + '/*/*.png'):
        if '-seg.png' in f or '-fseg.png' in f:
            continue
        folder_name = f.split('/')[-2]
        img_name = f.split('/')[-1].replace('.png', '')
        fn.write(folder_name + ' ' + img_name + '\n')
    fn.close()
