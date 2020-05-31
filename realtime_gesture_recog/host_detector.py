# Written by Vincent Xue
# Copyright (c) 2020 Vincent Xue

import socket
import sys
import re
import threading
import subprocess
import time

from utils import log_decorator


class HostDetector:
    def __init__(self, base_ip=None):
        self.hosts = {}
        if base_ip is None:
            self._base_ip = self.find_local_IP()
        else:
            self._base_ip = base_ip
        self._IPs = []

    @log_decorator
    def find_local_IP(self):
        '''
        use UDP to find local IP in public network.
        author: https://www.chenyudong.com/archives/python-get-local-ip-graceful.html#_UDP_IP
        :return:
        '''
        local_IP = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
                    [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
        print(">>> Local IP found: ",local_IP)
        return local_IP

    def set_base_IP(self, ip):
        self._base_ip = ip

    def get_IPs(self):
        return self._IPs

    def is_IP_used(self, ip):
        '''
        check whether the ip is used
        '''
        cmd_str = "ping {0} -n 1 -w 600".format(ip)
        DETACHED_PROCESS = 0x00000008  # dont create CMD windows
        try:
            subprocess.run(cmd_str, creationflags=DETACHED_PROCESS, check=True)  # only for windows
        except subprocess.CalledProcessError as err:
            pass  # ip not used
        else:
            self._IPs.append(ip)

    def detect_IPs(self):
        '''
        use multi-thread to detect used IPs.
        :return:
        '''
        start_ip = self._base_ip.split(".")
        start_ip[3] = '0'
        end_ip = start_ip.copy()
        end_ip[3] = '255'
        tmp_ip = start_ip

        pthread_list = []
        for i in range(int(start_ip[3]), int(end_ip[3]) + 1):
            tmp_ip[3] = str(i)
            ip = '.'.join(tmp_ip)
            pthread_list.append(threading.Thread(target=self.is_IP_used, args=(ip,)))
        for item in pthread_list:
            item.setDaemon(True)
            item.start()

    def get_host_name(self, ip):
        try:
            name, _, _ = socket.gethostbyaddr(ip)
        except:
            # ip doesnt have host name
            name = None
        if name is not None:
            self.hosts[name] = ip

    @log_decorator
    def detect_hosts(self):
        '''
        use multi-thread to detect used IPs.
        :return:
        '''
        if len(self._IPs) == 0:
            self.detect_IPs()
            time.sleep(0.5) # let sub threads run
        pthread_list = []
        for ip in self._IPs:
            pthread_list.append(threading.Thread(target=self.get_host_name, args=(ip,)))
        for item in pthread_list:
            item.setDaemon(True)
            item.start()


if __name__ == '__main__':
    detector = HostDetector()
    # detector.detect_IPs()
    # print(detector.get_IPs())
    detector.detect_hosts()
    # detector.get_host_name('192.168.43.111')
    time.sleep(5)
    print(detector.hosts)
