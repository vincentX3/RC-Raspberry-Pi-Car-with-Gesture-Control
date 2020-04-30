import cv2
import numpy as np
import copy
import math
from utils import PATH_DATA, PATH_DATA_TEST

# parameters
cap_region_x_begin = 0.6  # start point/total width
cap_region_y_end = 0.7  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0  # bool, whether the background captured


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


if __name__ == '__main__':
    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    capturing_type = ''
    capturing_count = {str(i): 0 for i in range(1, 7)}
    test_count = 0

    print('>>> open camera.')

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            cv2.imshow('mask', img)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print('>>> Background Captured')
        elif k == ord('s'):
            print('>>> Save image')
            img_name = PATH_DATA + '/' + capturing_type + '_' + str(capturing_count[capturing_type]) + '.png'
            cv2.imwrite(img_name, img)
            print('$saving: %s-%d' % (capturing_type, capturing_count[capturing_type]))
            capturing_count[capturing_type] += 1
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print('>>> Reset BackGround')
        elif k > ord('0') and k < ord('7'):
            # choose capturing class
            capturing_type = str(k - ord('0'))
            print('>>> Changing type to :', capturing_type)
        elif k == ord('t'):
            # capture test imgs
            print('>>> Save test image')
            img_name = PATH_DATA_TEST + '/' + str(test_count) + '.png'
            cv2.imwrite(img_name, img)
            test_count += 1
