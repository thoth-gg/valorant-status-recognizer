import cv2
import os
import numpy as np
import requests
from time import time, sleep

camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

result = np.zeros((10, 5), np.uint8)
latest = np.zeros((10), np.uint8)

akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

agents = {}

def init_agents():
    agentFiles = os.listdir('agents')
    for file in agentFiles:
        agent = cv2.imread('agents/' + file, cv2.IMREAD_GRAYSCALE)
        _, des = akaze.detectAndCompute(agent, None)
        agents[file.split('.')[0]] = des
        print("Agent loaded: " + file, len(des))

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
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskBGR =  cv2.inRange(hsv_image, (0,   110, 100), (3,   170, 255))
    maskBGR += cv2.inRange(hsv_image, (177, 110, 100), (180, 170, 255))
    return cv2.bitwise_and(frame, frame, mask = maskBGR)

framerate = 5
oneFrameTime = 1 / framerate

print("Loading agents...")
init_agents()
print("Agents loaded.")

while True:
    startAt = time()
    ret, frame = camera.read(0)
    extracted_frame = cv2.add(white_image(frame), red_image(frame))

    cv2.imshow('agent', extracted_frame)
    current = []
    for i in range(0, 5*66, 66):
        hpPercentage = get_hp(extracted_frame[79:80,448 + i:484 + i])
        current.append([hpPercentage])
    for i in range(0, 5*66, 66):
        hpPercentage = get_hp(extracted_frame[79:80,1173 + i:1211 + i])
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
    y = 85

    for i in range(0, 5):
        agent_frame = cv2.cvtColor(frame[31:68,446 + i * 66:486 + i * 66], cv2.COLOR_BGR2GRAY)
        kp, des = akaze.detectAndCompute(cv2.resize(agent_frame, (40 * 3, 37 * 3)), None)

        if(des is None): continue

        maxPoint = {"point": 0, "name": "Unknown"}
        for agent in agents:
            matches = bf.knnMatch(agents[agent], des, k=2)

            p = []
            for _, pair in enumerate(matches):
                try:
                    m, n = pair
                    if m.distance < 0.5 * n.distance:
                        p.append(m)
                except ValueError:
                    pass

            if(len(p) > maxPoint['point']):
                maxPoint = {"point": len(p), "name": agent}

        if(maxPoint['point'] == 0): continue
        cv2.putText(frame, maxPoint['name'], (446 + i * 66 + 4, y - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))

        cv2.line(frame, (448 + i * 66, y), (448 + round(35 * median[i]) + i * 66, y), (255, 255, 0), 2, cv2.LINE_4)
        cv2.putText(frame, str(round(median[i] * 100)), (446 + i * 66 + 4, y - 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))

    # for i in range(0, 5):
    #     cv2.line(extracted_frame, (1171 + i * 66, y), (1171 + round(40 * median[i + 5]) + i * 66, y), (255, 255, 0), 2, cv2.LINE_4)
    #     cv2.putText(extracted_frame, str(round(median[i + 5] * 100)), (1171 + i * 66 + 4, y - 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))

    cv2.imshow('frame', frame)

    res = cv2.waitKey(1) & 0xFF
    if res == ord('q'):
        break
