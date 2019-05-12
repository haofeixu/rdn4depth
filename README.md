## rdn4depth

A new learning based method to estimate depth from unconstrained monocular videos without ground truth supervision. The core contribution lies in Region Deformer Networks (RDN) for modeling various forms of object motions by the bicubic function. More details can be found in our paper:

>[Region Deformer Networks for Unsupervised Depth Estimation from Unconstrained Monocular Videos](https://arxiv.org/abs/1902.09907)
>
>[Haofei Xu](https://github.com/haofeixu), [Jianmin Zheng](http://www.ntu.edu.sg/home/asjmzheng/), [Jianfei Cai](http://www.ntu.edu.sg/home/asjfcai/) and [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/)
>
>[IJCAI 2019](https://www.ijcai19.org)

<p align="center"><img width=60% src="https://github.com/haofeixu/rdn4depth/blob/master/assets/demo.gif"></p>

Any questions or discussions are welcomed!

## RDN

The parameters of the bicubic function are learned by our proposed Region Deformer Network (RDN).

<p align="center"><img width=60% src="https://github.com/haofeixu/rdn4depth/blob/master/assets/rdn.png"></p>

## Installation

The code is developed with Python 3.6 and TensorFlow 1.2.0. For conda users ([anaconda](https://www.anaconda.com/distribution/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)), we have provided an `environment.yml` file, you can install with the following command:

```shell
conda env create -f environment.yml
```

## Data Preparation

### KITTI

You need to download [KITTI raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) first, then the raw data is processed with the following three steps:

1. Generate training data

   ```shell
   python prepare.py \
   --dataset_dir /path/to/kitti/raw/data \
   --dump_root /path/to/save/processed/data \
   --gen_data
   ```

2. Instance segmentation

   When training with our motion model, the instance segmentation mask is needed. We use  [an open source Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) to generate the segmentation mask. The raw output by Mask R-CNN is saved as lossless `.png` format (with shape [H, W, 3], the same value is used for all three channels, 0 for background and 1-255 for different instances). We name the raw output as `X-raw-fseg.png`, e.g. for image file `test.png`, you should save its segmentation as `test-raw-fseg.png`. 

3. Align segments across frames

   As the raw segments are not temporally consistent, we need to align them to make the same object have same instance id across frames.

   ```shell
   python prepare.py \
   --dump_root /path/to/processed/data \
   --align_seg
   ```

### Cityscapes

You need to download image sequence `leftImg8bit_sequence_trainvaltest.zip`  and calibration file `camera_trainvaltest.zip`  from [Cityscapes website](https://www.cityscapes-dataset.com/downloads/) (registration is needed to download the data), then the data is processed with the following three steps:

- Generate training data

  ```shell
  python prepare.py \
  --dataset cityscapes \
  --dataset_dir /path/to/cityscapes/data \
  --dump_root /path/to/save/processed/data \
  --gen_data
  ```

- Instance segmentation

  Same as KITTI.

- Align segments across frames

  ```shell
  python prepare.py \
  --dataset cityscapes \
  --dump_root /path/to/processed/data \
  --align_seg
  ```


## Training

Detailed training commands for reproducing our results are provided below. Every time you run, the command and flags will be saved to `checkpoint_dir/command.txt` and `checkpoint_dir/flags.json` to track experiments history.

### KITTI

- Baseline

  ```shell
  python train.py \
  --logtostderr \
  --checkpoint_dir checkpoints/kitti-baseline \
  --data_dir /path/to/processed/kitti/data \
  --imagenet_ckpt /path/to/pretrained/resnet18/model \
  --seg_align_type null
  ```

- Motion

  ```shell
  python train.py \
  --logtostderr \
  --checkpoint_dir checkpoints/kitti-motion \
  --data_dir /path/to/processed/kitti/data \
  --handle_motion \
  --pretrained_ckpt /path/to/pretrained/baseline/model \
  --learning_rate 2e-5 \
  --object_depth_weight 0.5
  ```

### Cityscapes

- Baseline

  ```shell
  python train.py \
  --logtostderr \
  --checkpoint_dir checkpoints/cityscapes-baseline \
  --data_dir /path/to/processed/cityscapes/data \
  --imagenet_ckpt /path/to/pretrained/resnet18/model \
  --seg_align_type null \
  --smooth_weight 0.008
  ```

- Motion

  ```shell
  python train.py \
  --logtostderr \
  --checkpoint_dir checkpoints/cityscapes-motion \
  --data_dir /path/to/processed/cityscapes/data \
  --handle_motion \
  --pretrained_ckpt /path/to/pretrained/baseline/model \
  --learning_rate 2e-5 \
  --object_depth_weight 0.5 \
  --object_depth_threshold 0.5 \
  --smooth_weights 0.008
  ```

## Models

The trained models for KITTI and Cityscapes dataset are available at [Google Drive](https://drive.google.com/file/d/1y1uFP_tKzHaJr5Rh28rKkMw2xDeSgSAN/view?usp=sharing).

## Inference

Inference can be running on an image list file (for evaluation) or an image directory (for visualization).

### KITTI

```shell
python inference.py \
--logtostderr \
--depth \
--input_list_file dataset/test_files_eigen.txt \
--output_dir output/ \
--model_ckpt /path/to/trained/model/ckpt
```

### Cityscapes

```shell
python inference.py \
--logtostderr \
--depth \
--input_dir /path/to/cityscapes/data/directory \
--output_dir output/cityscapes \
--model_ckpt /path/to/trained/model/ckpt \
--not_save_depth_npy \
--inference_crop cityscapes
```

## Evaluation

You can use the `pack_pred_depths` function in `utils.py` to generate a single depth prediction file for evaluation. We also make our depth prediction results on KITTI Eigen test split available at [Google Drive](https://drive.google.com/open?id=1f6UZnewTXsymABSbpZjFzUGoj7kg809I).

###  On the whole image

Standard evaluation protocol on KITTI Eigen test split to compare with previous methods.

```shell
python evaluate.py \
--kitti_dir /path/to/kitti/raw/data \
--pred_file /path/to/depth/prediction/file
```

### On specific objects

We also evaluate on specific objects to highlight the performance gains brought by our proposed RDN, which is realized by using the segmentation masks from Mask R-CNN. The segmentation masks for people and cars used in our paper are available at [Google Drive](https://drive.google.com/open?id=1Rw5y1nNOK9RDg-fmkMZ8HVQEw4GPDJFx).

```shell
python evaluate.py \
--kitti_dir /path/to/kitti/raw/data \
--pred_file /path/to/depth/prediction/file \
--mask people \
--seg_dir /path/to/eigen/test/split/segments
```

The evaluation results on people and cars of KITTI Eigen test split are as follows. If you want to compare with our results, please make sure to use the same object segmentation masks with us.

<p align="center"><img width=90% src="https://github.com/haofeixu/rdn4depth/blob/master/assets/eval_cars_people.png"></p>

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{xu2019rdn4depth,
  title={Region Deformer Networks for Unsupervised Depth Estimation from Unconstrained Monocular Videos},
  author={Xu, Haofei and Zheng, Jianmin and Cai, Jianfei and Zhang, Juyong},
  booktitle={IJCAI},
  year={2019}
}
```

## Acknowledgements

The code is inspired by [struct2depth](https://github.com/tensorflow/models/tree/master/research/struct2depth), we thank Vincent Casser and Anelia Angelova for clarifying the details of their work.