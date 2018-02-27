#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math
import signal
import random
import time
from multiprocessing import Process, Queue, Event

from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper

exitEvent = Event() # for noitfy all process exit.

def handler(sig_num, stack_frame):
    global exitEvent
    exitEvent.set()
signal.signal(signal.SIGINT, handler)

class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._process_list = []
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=3)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        self._max_epoch = kwargs['max_epoch']
        # start buffering
        self._start_buffering(kwargs['input_paths'], kwargs['landmark_type'])

    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    datas = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield datas

    def get_epoch(self):
        return self._curr_epoch

    def should_stop(self):
        if exitEvent.is_set() or self._curr_epoch > self._max_epoch:
            exitEvent.set()
            self.__clear_and_exit()
            return True
        else:
            return False

    def __clear_and_exit(self):
        print ("[Exiting] Clear all queue.")
        while True:
            time.sleep(1)
            _alive = False
            for i in range(self._process_num):
                try:
                    self._output_queue[i].get(block=True, timeout=0.01)
                    _alive = True
                except Exception as ex:
                    pass
            if _alive == False: break
        print ("[Exiting] Confirm all process is exited.")
        for i in range(self._process_num):
            if self._process_list[i].is_alive():
                print ("[Exiting] Force to terminate process %d"%(i))
                self._process_list[i].terminate()
        print ("[Exiting] Batch reader clear done!")

    def _start_buffering(self, input_paths, landmark_type):
        if type(input_paths) in [str, unicode]:
            input_paths = [input_paths]
        for input_path in input_paths:
            for line in open(input_path):
                info = LandmarkHelper.parse(line, landmark_type)
                self._sample_list.append(info)
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            self._process_list.append(p)

    def _process(self, idx, sample_list):
        __landmark_augment = LandmarkAugment()
        # read all image to memory to speed up!
        if self._kwargs['buffer2memory']:
            print ("[Process %d] Start to read image to memory! Count=%d"%(idx, len(sample_list)))
            sample_list = __landmark_augment.mini_crop_by_landmarks(
                sample_list, LandmarkHelper.get_scales(self._kwargs['landmark_type'])[1], self._kwargs['img_format'])
            print ("[Process %d] Read all image to memory finish!"%(idx))
        sample_cnt = 0 # count for one batch
        image_list, landmarks_list, angle_list = [], [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                if type(sample[0]) in [str, unicode]:
                    image = cv2.imread(sample[0])
                    if self._kwargs['img_format'] == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imdecode(sample[0], cv2.IMREAD_COLOR)
                landmarks = sample[1].copy()# keep deep copy
                scale_range = LandmarkHelper.get_scales(self._kwargs['landmark_type'])[0]
                image_new, landmarks_new = __landmark_augment.augment(image, landmarks, self._kwargs['img_size'],
                                            self._kwargs['max_angle'], scale_range)
                # sent a batch
                sample_cnt += 1
                image_list.append(image_new)
                landmarks_list.append(landmarks_new)
                if self._kwargs.get("train_angle"):
                    angle_list.append(sample[2])
                if sample_cnt >= self._kwargs['batch_size']:
                    if self._kwargs.get("train_angle"):
                        datas = (np.array(image_list), np.array(landmarks_list),
                                 np.array(angle_list))
                    else:
                        datas = (np.array(image_list), np.array(landmarks_list))
                    self._output_queue[idx].put(datas)
                    sample_cnt = 0
                    image_list, landmarks_list, angle_list = [], [], []
                # if exit
                if exitEvent.is_set():
                    break
            if exitEvent.is_set():
                break
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0

# use for unit test
if __name__ == '__main__':
    kwargs = {
        #'input_paths': "/world/data-c9/liubofang/dataset_original/CelebA/full_path_zf_bbox_pts.txt",
        #'landmark_type': 5,
        # 'input_paths': "/world/data-c9/liubofang/dataset_original/CelebA/CelebA_19w_83points_bbox_angle.txt",
        # 'landmark_type': 83,
        'input_paths': "/home/liubofang/other_script/face_landmark_人脸关键点训练集创建/results_new_bbox_list.json",
        'landmark_type': 72,
        'batch_size': 512,
        'process_num': 2,
        'img_format': 'RGB',
        'img_size': 128,
        'max_angle': 10,
        'max_epoch':1,
        'train_angle': True,
        'buffer2memory': False,
    }
    b = BatchReader(**kwargs)
    g = b.batch_generator()
    output_folder = "output_tmp/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    import time
    start_time = time.time()
    while not b.should_stop():
        end_time = time.time()
        print ("get new batch...epoch: %d. cost: %.3f"%(
                b.get_epoch(), end_time-start_time))
        start_time = end_time
        datas = g.next()
        for idx, image in enumerate(datas[0]):
            if kwargs.get("train_angle"):
                angles = datas[2][idx]
                print "idx = {} with angle {}".format(idx, angles)
            landmarks = datas[1][idx]
            if idx > 20: # only see first 10
                break
            landmarks = landmarks.reshape([-1, 2])
            # image = cv2.resize(image, (1080, 1080)) # for debug
            # for i, l in enumerate(landmarks):
                # ii = tuple(l * image.shape[:2])
                # cv2.circle(image, (int(ii[0]), int(ii[1])), 1, (0,255,0), -1)
                # cv2.putText(image, str(i), (int(ii[0]), int(ii[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.imwrite("%s/%d.jpg"%(output_folder, idx), image)
        break

