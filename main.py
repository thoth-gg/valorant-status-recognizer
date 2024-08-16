import cv2
import os
import numpy as np
import requests
from time import time, sleep

cameraList = [
    cv2.VideoCapture(1),
    cv2.VideoCapture(0),
]
for camera in cameraList:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

result = np.zeros((10, 5), np.uint8)
latest = {}

akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

agents = {}

def init_agents():
    agentFiles = os.listdir('agents')
    for file in agentFiles:
        if file == '.gitkeep': continue
        agent = cv2.imread('agents/' + file, cv2.IMREAD_GRAYSCALE)
        _, des = akaze.detectAndCompute(agent, None)
        agents[file.split('.')[0]] = des
        print("Agent loaded: " + file, len(des))

def white_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)

def red_image(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask =  cv2.inRange(hsv_image, (0,   110, 100), (3,   170, 255))
    mask += cv2.inRange(hsv_image, (177, 110, 100), (180, 170, 255))
    return cv2.bitwise_and(frame, frame, mask = mask)

def get_hp(frame):
    binary_frame = cv2.add(white_image(frame), red_image(frame))
    gray = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    pixel_number = np.size(img_binary)
    pixel_sum = np.sum(img_binary)
    white_pixel_number = pixel_sum / 255
    return white_pixel_number / pixel_number

framerate = 5
oneFrameTime = 1 / framerate

print("Loading agents...")
init_agents()
print("Agents loaded.")

while True:
    startAt = time()

    current = []
    agent_frame_list = []
    for camera in cameraList:
        _, frame = camera.read()
        for i in range(0, 5):
            agent_frame_list.append(cv2.cvtColor(frame[31:68,446 + i * 66:486 + i * 66], cv2.COLOR_BGR2GRAY))
            hpPercentage = get_hp(frame[79:80,448 + i * 66:484 + i * 66])
            current.append([round(hpPercentage * 100)])

    result = np.delete(result, 0, 1)
    result = np.hstack([result, current])

    median = np.median(result, axis=1)

    data = {}
    for i in range(0, 5):
        kp, des = akaze.detectAndCompute(cv2.resize(agent_frame_list[i], (40 * 3, 37 * 3)), None)

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

        data[maxPoint['name']] = median[i]

    data = {}
    for i in range(0, 5):
        kp, des = akaze.detectAndCompute(cv2.resize(agent_frame_list[i], (40 * 3, 37 * 3)), None)

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

        data[maxPoint['name']] = median[i]

    if(latest != data):
        print(data)
        latest = data
        # requests.post('http://localhost:3000/hp', json=data)

    endAt = time()
    print("time: ", (endAt - startAt) * 1000, "ms")
    if endAt - startAt < oneFrameTime:
        sleep(oneFrameTime - (endAt - startAt))
