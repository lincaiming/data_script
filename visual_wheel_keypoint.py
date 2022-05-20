# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    catIds = coco.getCatIds(catNms=['person'])

    # dataset = pose_model.cfg.data['test']['type']
    # dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    # if dataset_info is None:
    #     warnings.warn(
    #         'Please set `dataset_info` in the config.'
    #         'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
    #         DeprecationWarning)
    # else:
    #     dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    WHEEL_COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 130, 255)]
    # process each image
    for frameid in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[frameid]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[1])
        img = cv2.imread(image_name)
        save_img = False
        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            # ann = coco.anns[ann_id]
            anns = coco.loadAnns(ann_id)
            for ann in anns:
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3].astype(np.int32)
                    y = kp[1::3].astype(np.int32)
                    v = kp[2::3].astype(np.int32)
                    print(v)
                    # for sk in sks:
                    #     if np.all(v[sk] > 0):
                    #          cv2.line(img, (x[sk][0], y[sk][0]), (x[sk][1], y[sk][1]), (255, 0, 0), 1)
                    for i in range(x.shape[0]):
                        if v[i] > 0:
                            cv2.putText(img, "{}".format(i+1), (x[i]+1, y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, WHEEL_COLOR[i], 1, 4)
                            cv2.circle(img, (x[i], y[i]), 2, WHEEL_COLOR[i], -1)
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                x1,  y1, x2, y2 = int(bbox_x), int(bbox_y), int(bbox_x + bbox_w), int(bbox_y + bbox_h)
                cv2.rectangle(img,(x1, y1), (x2, y2), (0, 0, 255), 1)
            save_img = True
            # coco.showAnns(anns)
        if save_img:
            print(frameid)
            cv2.imwrite(os.path.join(args.out_img_root, os.path.basename(image_name)), img)




if __name__ == '__main__':
    main()
