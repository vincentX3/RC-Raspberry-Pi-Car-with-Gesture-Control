# Written by Vincent Xue
# Copyright (c) 2020 Vincent Xue

import socket


class CarController:
    def __init__(self, ip, port=8888):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd = {
            'turn_left': '0',
            'turn_right': '1',
            'back_up': '2',
            'go_ahead': '3',
            'idle': '4'
        }
        self.response = None
        self.status = 'offline'
        self._status_type = ['offline', 'online', 'idle']

    def move(self, direction):
        '''
        send direction command to controller
        :param direction:
        :return:
        '''
        if direction in self.cmd:
            self.socket.sendto(self.cmd[direction].encode(), (self.ip, self.port))

    def test(self):
        '''
        check connection
        :return:
        '''
        pass
