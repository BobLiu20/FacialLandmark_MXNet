import os
import sys
import cv2
import numpy as np
import argparse

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'common'))
from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper
from predictor import Predictor

TEST_DATASET = {
    5: {"lfw_5590_net_7876": "/world/data-c9/liubofang/dataset_original/other2013/testImageList.txt"},
    83: {"celeba": "/world/data-c9/liubofang/dataset_original/CelebA/CelebA_19w_83points_bbox_angle.txt"},
}

def isDatasetReady(landmark_type, dataset_name):
    if TEST_DATASET.get(landmark_type, {}).get(dataset_name, None) == None:
        print ("Unsupport landmark_type or dataset_name.")
        print ("Current available is %s."%(str(TEST_DATASET)))
        return False
    if not os.path.exists(TEST_DATASET[landmark_type][dataset_name]):
        print ("The dataset path is not exist: %s"%(TEST_DATASET[landmark_type][dataset_name]))
        return False
    return True

def datasetReader(landmark_type, dataset_name):
    __landmark_augment = LandmarkAugment()
    if landmark_type == 5 and dataset_name == "lfw_5590_net_7876":
        txt_path = TEST_DATASET[landmark_type][dataset_name]
        for line in open(txt_path):
            line = line.split()
            path = os.path.join(os.path.dirname(txt_path), line[0])
            image = cv2.imread(path)
            if image is None:
                print "Skip: ", path
                continue
            # facePos = map(int, [line[1], line[3], line[2], line[4]]) # x1,y1,x2,y2
            landmarks = map(float, line[5:]) # x1,y1,...,x5,y5
            landmarks = np.array(landmarks, dtype=np.float32).reshape(5, 2)
            scale_range = LandmarkHelper.get_scales(landmark_type)[0]
            scale = (scale_range[0] + scale_range[1]) / 2.0
            (x1, y1, x2, y2), _, _, _ = __landmark_augment.get_bbox_of_landmarks(image, landmarks, scale, 0.5)
            yield image, landmarks, (x1, y1, x2, y2) # bbox
    elif landmark_type == 83 and dataset_name == "celeba":
        txt_path = TEST_DATASET[landmark_type][dataset_name]
        for line in open(txt_path):
            line = line.split()
            path = line[0]
            image = cv2.imread(path)
            if image is None:
                print "Skip: ", path
                continue
            landmarks = np.fromstring(line[1], dtype=float, count=166, sep=',')
            landmarks = landmarks.reshape((-1, 2))
            scale_range = LandmarkHelper.get_scales(landmark_type)[0]
            scale = (scale_range[0] + scale_range[1]) / 2.0
            (x1, y1, x2, y2), _, _, _ = __landmark_augment.get_bbox_of_landmarks(image, landmarks, scale, 0.5)
            yield image, landmarks, (x1, y1, x2, y2) # bbox
    else:
        raise Exception("No reader for %d and %d"%(landmark_type, dataset_name))

def mseNormlized(ground_truth, pred, landmark_type):
    ground_truth = ground_truth.reshape((-1, 2))
    pred = pred.reshape((-1, 2))
    if landmark_type == 5:
        eyeDistance = np.linalg.norm(ground_truth[0]-ground_truth[1])
    elif landmark_type == 83:
        eyeDistance = np.linalg.norm(ground_truth[79]-ground_truth[81])
    else:
        raise Exception("Unsupport landmark type.")
    norm_mean = np.linalg.norm(pred - ground_truth, axis=1).mean()
    if eyeDistance > 0.0:
        return (norm_mean / eyeDistance)  # normlized 
    else:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model', type=str, default='fanet8ss_inference')
    parser.add_argument('--landmark_type', type=int, default=5) #5 or 83
    parser.add_argument('--dataset_name', type=str, default="lfw_5590_net_7876")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--img_format', type=str, default='RGB')
    parser.add_argument('--gpu_device', type=str, default='1')
    parser.add_argument('--debug', type=bool, default=False)
    arg_dict = vars(parser.parse_args())

    if not isDatasetReady(arg_dict['landmark_type'], arg_dict['dataset_name']):
        print ("Exit now. Bye!")
        sys.exit()

    predictor = Predictor(arg_dict)
    errorPerLandmark = 0.0
    numCount = 0
    failureCount = 0
    print "Processing:"
    for image, landmarks, bbox in datasetReader(arg_dict['landmark_type'], arg_dict['dataset_name']):
        landmarks[:,0] = (landmarks[:, 0] - bbox[0]) / float(bbox[2]-bbox[0])
        landmarks[:,1] = (landmarks[:, 1] - bbox[1]) / float(bbox[3]-bbox[1])

        faceImage = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        prediction = predictor.predict([faceImage])[0]
        prediction = prediction.reshape([-1, 2])
        normlized = mseNormlized(landmarks, prediction, arg_dict['landmark_type'])
        errorPerLandmark += normlized
        numCount += 1
        if normlized > 0.1: # Count error above 0.1 as failure
            failureCount += 1
            if arg_dict['debug'] and failureCount < 50:
                for (x, y) in prediction:
                    y = int(y * arg_dict['img_size']) 
                    x = int(x * arg_dict['img_size']) 
                    cv2.circle(faceImage, (x, y), 1, (0, 255, 0), -1)
                cv2.imwrite("output_tmp/benchmark_fail_%d.jpg"%failureCount, faceImage)
        print "\r%d"%(numCount),
        sys.stdout.flush()

    meanError = errorPerLandmark/numCount
    print "\nResult: \nmean error: %.05f \nfailures(err>0.1): %.02f%%(%d/%d)"%(meanError, 
        (failureCount/float(numCount)*100.0), failureCount, numCount)

