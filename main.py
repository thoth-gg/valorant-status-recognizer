import cv2
import numpy as np
import requests
from time import time, sleep

camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

result = np.zeros((10, 10), np.uint8)
latest = np.zeros((10), np.uint8)

def get_hp(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    pixel_number = np.size(img_binary)
    pixel_sum = np.sum(img_binary)
    white_pixel_number = pixel_sum / 255
    return white_pixel_number / pixel_number

def white_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)

def red_image(frame):
    bgr = [86,93,232]
    thresh = 50
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
    maskBGR = cv2.inRange(frame,minBGR,maxBGR)
    return cv2.bitwise_and(frame, frame, mask = maskBGR)

framerate = 5
oneFrameTime = 1 / framerate

while True:
    startAt = time()
    ret, frame = camera.read(0)
    extracted_frame = cv2.add(white_image(frame), red_image(frame))

    current = []
    for i in range(0, 5*66, 66):
        hpPercentage = get_hp(extracted_frame[79:80,446 + i:486 + i])
        current.append([hpPercentage])
    for i in range(0, 5*66, 66):
        hpPercentage = get_hp(extracted_frame[79:80,1171 + i:1211 + i])
        current.append([hpPercentage])
    result = np.delete(result, 0, 1)
    result = np.hstack([result, current])

    median = np.median(result, axis=1)

    if((latest != median).any()):
        print(median)
        latest = median
        data = {
            "hp": median.tolist()
        }
        requests.post('http://localhost:3000/hp', json=data)

    endAt = time()

    if endAt - startAt < oneFrameTime:
        sleep(oneFrameTime - (endAt - startAt))
