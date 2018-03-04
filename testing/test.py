#coding=utf8
import os
import sys
import argparse
import cv2
import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet import gluon

from predictor import Predictor

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'training'))
import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model', type=str, default='fanet8ss_inference')
    parser.add_argument('--landmark_type', type=int, default=5) #5 or 72 or 83
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--images_folder', type=str, default='test_images')
    parser.add_argument('--img_format',type=str,default='RGB')
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--train_angle', type=bool, default=False, help="angle loss.")
    arg_dict = vars(parser.parse_args())
    # output folder
    out_folder = os.path.join(MY_DIRNAME, "output_tmp")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    # predict
    predictor = Predictor(arg_dict)
    for idx, im_path in enumerate(os.listdir(arg_dict['images_folder'])):
        im_path = os.path.join(arg_dict['images_folder'], im_path)
        img = cv2.imread(im_path)
        h, w = img.shape[:2]
        prediction = predictor.predict([img])
        if arg_dict['train_angle']:
            points, angles = prediction[0][0], prediction[1][0]
        else:
            points, angles = prediction[0], None
        for (x, y) in points.reshape([-1, 2]):
            y = int(y * h) 
            x = int(x * w) 
            cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(out_folder, os.path.basename(im_path)), img)
        if angles is not None:
            print ("predicting {}, angle {}".format(im_path, angles.tolist()))
        else:
            print ("predicting {}".format(im_path))

