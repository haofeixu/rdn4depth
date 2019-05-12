import argparse

from dataset.gen_data_kitti import gen_data_kitti
from dataset.gen_data_cityscapes import gen_data_cityscapes
from dataset.align import align_segs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='kitti', type=str, choices=['kitti', 'cityscapes'],
                    help='Dataset name, kitti or cityscapes')
parser.add_argument('--dataset_dir', default=None, help='Path to raw dataset')
parser.add_argument('--dump_root', default=None, required=True, help='Path to save processed data')
parser.add_argument('--img_height', default=128, type=int, help='Image height')
parser.add_argument('--img_width', default=416, type=int, help='Image width')
parser.add_argument('--seq_length', default=3, type=int, help='Sequence length')
parser.add_argument('--gen_data', action='store_true', help='Generate data')
parser.add_argument('--align_seg', action='store_true', help='Align segments across frames')

args = parser.parse_args()


def main():
    if args.dataset == 'kitti':
        if args.gen_data:
            gen_data_kitti(args.dataset_dir, args.dump_root, args.img_height,
                           args.img_width, args.seq_length)
        elif args.align_seg:
            # Assume segmentation is done for processed data and
            # segments are saved to the same path of processed data
            align_segs(args.dump_root)

    elif args.dataset == 'cityscapes':
        if args.gen_data:
            gen_data_cityscapes(args.dataset_dir, args.dump_root, args.img_height,
                                args.img_width, args.seq_length)
        elif args.align_seg:
            align_segs(args.dump_root)


if __name__ == '__main__':
    main()
