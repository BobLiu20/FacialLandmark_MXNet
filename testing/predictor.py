#coding=utf8
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet import gluon

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'training'))
import models

class Predictor():
    def __init__(self, _arg_dict):
        self._arg_dict = _arg_dict
        # set up environment
        os.environ['CUDA_VISIBLE_DEVICES'] = self._arg_dict['gpu_device']
        self.__init_model()

    def __init_model(self):
        self.net =  models.init(num_label=self._arg_dict['landmark_type'] * 2, **self._arg_dict)
        self.net.load_params(self._arg_dict['model_path'], ctx=mx.gpu())
        print self.net

    def preprocess(self, images):
        images_new = []
        for img in images:
            if self._arg_dict['img_format'] == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self._arg_dict['img_size'], self._arg_dict['img_size']))
            images_new.append(img)
        images_new = np.asarray(images_new)
        images_new = nd.array(images_new, ctx=mx.gpu())
        images_new = nd.transpose(images_new.astype('float32'), (0,3,1,2)) / 127.5 - 1.0
        return images_new

    def predict(self, images):
        images = self.preprocess(images)
        res = self.net(images)
        if self._arg_dict.get('train_angle'):
            return res[0].asnumpy(), res[1].asnumpy()
        else:
            return res.asnumpy()

