import cv2
import numpy as np

from utils import log_decorator

minValue = 70

kernel = np.ones((15, 15), np.uint8)
kernel2 = np.ones((1, 1), np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Which mask mode to use BinaryMask, SkinMask (True|False) OR BkgrndSubMask ('x' key)
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0

def skinMask(frame, x0, y0, width, height):
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    # cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0 + height, x0:x0 + width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res


def binaryMask(frame, x0, y0, width, height):

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0 + height, x0:x0 + width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res


def bkgrndSubMask(frame, x0, y0, width, height):
    global turn_on_prediction, takebkgrndSubMask, model, bkgrnd, saveImg

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Take background image
    if takebkgrndSubMask == True:
        bkgrnd = roi
        takebkgrndSubMask = False
        print("Refreshing background image for mask...")

    # Take a diff between roi & bkgrnd image contents
    diff = cv2.absdiff(roi, bkgrnd)

    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.GaussianBlur(diff, (3, 3), 5)
    mask = cv2.erode(diff, skinkernel, iterations=1)
    mask = cv2.dilate(diff, skinkernel, iterations=1)
    res = cv2.bitwise_and(roi, roi, mask=mask)


    return res
