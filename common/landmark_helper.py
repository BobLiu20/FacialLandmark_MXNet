# coding=utf-8
'''
Bob.Liu in 20171114
'''
import numpy as np
import cv2
import json

class LandmarkHelper(object):
    '''
    Helper for different landmark type
    '''
    @classmethod
    def parse(cls, line, landmark_type):
        '''
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input txt
            landmark_type: len of landmarks
        Return:
            see child parse
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            return cls.__landmark5_txt_parse(line)
        elif landmark_type == 83:
            return cls.__landmark83_txt_parse(line)
        elif landmark_type == 72:
            return cls.__landmark72_txt_parse(line)
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def flip(a, landmark_type):
        '''
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: len of landmarks
        Returns:
            landmarks: new landmarks
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            landmarks = np.concatenate((a[1,:], a[0,:], a[2,:], a[4,:], a[3,:]), axis=0)
        elif landmark_type == 72:
            landmarks = np.concatenate((a[7:13][::-1], a[6:7], a[0:6][::-1], a[30:35][::-1],
                a[35:38][::-1], a[38:39], a[39:44][::-1], a[44:47][::-1], a[13:18][::-1],
                a[18:21][::-1], a[21:22], a[22:27][::-1], a[27:30][::-1], a[52:57][::-1],
                a[47:52][::-1], a[57:58], a[58:63][::-1], a[63:66][::-1], a[66:69][::-1],
                a[69:72][::-1]), axis=0)
        elif landmark_type == 83:
            landmarks = np.concatenate((a[10:19][::-1], a[9:10], a[0:9][::-1], a[35:36],
                a[36:43][::-1], a[43:48][::-1], a[48:51][::-1], a[19:20], a[20:27][::-1],
                a[27:32][::-1], a[32:35][::-1], a[56:60][::-1], a[55:56], a[51:55][::-1],
                a[60:61], a[61:72][::-1], a[72:73], a[73:78][::-1], a[80:81], a[81:82],
                a[78:79], a[79:80], a[82:83]), axis=0)
        else:
            raise Exception("Unsupport landmark type...")
        return landmarks.reshape([-1, 2])

    @staticmethod
    def get_scales(landmark_type):
        '''
        use for scales bbox according to bbox of landmarks
        Args:
            landmark_type: len of landmarks
        Returns:
            (min, max), min crop
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            return (2.7, 3.3), 4.5
        elif landmark_type in [72, 83]:
            return (1.2, 1.5), 2.6
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def __landmark5_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=[0:4] is bbox and [4:] is landmarks
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        data = map(int, a[1:])
        pts = data[4:] # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark83_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks83, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        a1 = np.fromstring(a[1], dtype=int, count=166, sep=',')
        a1 = a1.reshape((-1, 2))
        return a[0], a1

    @staticmethod
    def __landmark72_txt_parse(line):
        '''
        Args:
            line: a dict. 
            limit angle to +-90
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        d = json.loads(line)
        path = d["path"]
        landmarks = np.asarray([[int(v["x"]), int(v["y"])] for v in d["landmark"]], dtype=int)
        angle = np.clip((np.asarray([d["pitch"], d["yaw"], d["roll"]]) + 90.0) / 180.0, 0.0, 1.0)
        return path, landmarks, angle

