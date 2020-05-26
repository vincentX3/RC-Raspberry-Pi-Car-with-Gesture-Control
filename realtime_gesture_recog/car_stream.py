import urllib3
import numpy as np
import cv2

from utils import log_decorator, RASPBERRY_IP_PORT


class CarStream:
    def __init__(self,host=RASPBERRY_IP_PORT):
        self.host = host
        self.stream_address = 'http://' + host + '/?action=stream'
        # self.stream_address = 'rtmp://58.200.131.2:1935/livetv/hunantv' # for test
        self.buffer = None
        self.stream = None

    # @log_decorator
    def pull(self):
        '''
        pull a frame from stream and store in buffer
        :return:
        '''
        if self.stream is None:
            try:
                self.stream =  cv2.VideoCapture(self.stream_address)
            except Exception as e:
                print(">>> ERROR: can't open car video stream. Details:\n%s"%e)
        if self.stream and self.stream.isOpened():
            _, self.buffer=self.stream.read()
        else:
            print(">>> ERROR: can't pull frame from stream.")

    # @log_decorator
    def read_buffer(self,update=True):
        '''
        return frame stored in buffer, if update sets True then update the buffer.
        :return:
        '''
        if self.buffer is not None:
            frame = self.buffer.copy()
        else:
            print('>>> WARNING: empty buffer.')
            frame = None
        if update:
           self.pull()
        return frame

