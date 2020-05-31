# Written by Vincent Xue
# Copyright (c) 2020 Vincent Xue

from functools import wraps
import cv2
import numpy as np

# MACRO
PATH_DATA = './data/train'
PATH_DATA_TEST = './data/test'
PATH_MODELS = './weights'
MODEL_VERSION = '/naiveNN_v1'
RASPBERRY_IP_PORT = '192.168.43.97:8080'

C_NUM = 6  # total number of classes
C_TURN_LEFT = 1
C_TURN_RIGHT = 2
C_BACK_UP = 3
C_GOAHEAD = 4
C_IDLE = 5
C_NOTHING = 6
GESTURE_CLASSES = ('turn_left', 'turn_right', 'back_up', 'go_ahead', 'idle', 'nothing')

FONT = cv2.FONT_HERSHEY_SIMPLEX
SIZE = 0.5
FX = 10
FY = 350
FH = 18


def log_decorator(func):
    @wraps(func)
    def log(*args, **kwargs):
        try:
            print(">>> Executing:", func.__name__)
            return func(*args, **kwargs)
        except Exception as e:
            print(">>> Error: %s" % e)

    return log
