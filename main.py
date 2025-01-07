import cv2
import os
import numpy as np
import requests
from time import time, sleep
from configupdater import ConfigUpdater

cfg = ConfigUpdater()
cfg.read("config.ini")

framerate = int(cfg["common"]["framerate"].value)
report_url = cfg["common"]["report_url"].value
oneFrameTime = 1 / framerate

white_threshold = []
white_threshold.append(int(cfg["threshold0"]["white"].value))
white_threshold.append(int(cfg["threshold1"]["white"].value))
red_threshold_center = []

debug = cfg["common"]["debug"].value == "true"

# カメラ初期化
cameraList = [
    cv2.VideoCapture(0, cv2.CAP_DSHOW),
    cv2.VideoCapture(1, cv2.CAP_DSHOW),
]
for camera in cameraList:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Akaze初期化
akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# エージェント初期化
agents = {}
print("Loading agents...")
agent_files = os.listdir('agents')
for file in agent_files:
    if file == '.gitkeep': continue
    agent = cv2.imread('agents/' + file, cv2.IMREAD_GRAYSCALE)
    _, des = akaze.detectAndCompute(agent, None)
    agents[file.split('.')[0]] = des
    print("Agent loaded: " + file, len(des))
agent_list = list(agents.keys())
print("Agents loaded.")

def white_image(frame, i):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, white_threshold[i], 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)

def red_image(frame, i):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask =  cv2.inRange(hsv_image, (0,   110, 100), (3,   170, 255))
    mask += cv2.inRange(hsv_image, (177, 110, 100), (180, 170, 255))
    return cv2.bitwise_and(frame, frame, mask = mask)

def get_hp(frame, i):
    binary_frame = cv2.add(white_image(frame, i), red_image(frame, i))
    gray = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
    _, img_binary= cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    pixel_number = np.size(img_binary)
    pixel_sum = np.sum(img_binary)
    white_pixel_number = pixel_sum / 255
    return white_pixel_number / pixel_number

def detect_agent(frame):
    _, des = akaze.detectAndCompute(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (40 * 3, 37 * 3)), None)
    if (des is None): return None

    max_point = {"point": 0, "name": "Unknown"}
    for agent in agents:
        matches = bf.knnMatch(agents[agent], des, k=2)
        p = 0
        for _, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.5 * n.distance:
                    p += 1
            except ValueError:
                pass
        if(p > max_point['point']):
            max_point = {"point": p, "name": agent}
    if(max_point['point'] == 0): return None

    return max_point['name']

history = [[], []]
latest = [[], []]
current = [[], []]
for i in range(0, 2):
    history[i] = np.zeros((len(agents), 5), np.float64)
    latest[i] = np.zeros((len(agents)), np.float64)
    current[i] = np.zeros((len(agents)), np.float64)

while True:
    startAt = time()

    detected_agent_list = [[], []]
    debug_ui_list = []

    for i in range(0, 2):
        history[i] = np.delete(history[i], 0, 1)
        history[i] = np.hstack([history[i], np.zeros((len(agents), 1), np.float64)])
    
    for i, camera in enumerate(cameraList):
        _, frame = camera.read()
        if (debug):
            f = frame[0:100,400:1920-400]
            w = white_image(f, i)
            r = red_image(f, i)
            debugUi = cv2.vconcat([f, w, r, cv2.add(w, r)])
            if (i == 0):
                cv2.putText(debugUi, "Source", (4, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "White / Value=" + str(white_threshold[i]) + " -=[1] +=[3]", (4, 116), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "Red / Value=" + str(red_threshold_center), (4, 216), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "Result / Save=[Enter] Quit=[ESC]", (4, 316), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
            if (i == 1):
                cv2.putText(debugUi, "Source", (4, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "White / Value=" + str(white_threshold[i]) + " -=[7] +=[9]", (4, 116), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "Red / ", (4, 216), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(debugUi, "Result / Save=[Enter] Quit=[ESC]", (4, 316), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
            debug_ui_list.append(debugUi)
        
        for j in range(0, 5):
            agent_name = detect_agent(frame[31:68,446 + j * 66:486 + j * 66])
            if(agent_name is None): continue
            detected_agent_list[i].append(agent_name)
            hpPercentage = get_hp(frame[79:80,448 + j * 66:484 + j * 66], i)
            history[i][agent_list.index(agent_name)][-1] = round(hpPercentage * 100)

    for i in range(0, 2):
        current[i] = np.median(history[i], axis=1)

    if((latest[0] != current[0]).any() or (latest[1] != current[1]).any()):
        latest[0] = np.copy(current[0])
        latest[1] = np.copy(current[1])
        if (report_url != ""):
            data = [{}, {}]

            for i in range(0, 2):
                for agent_name in detected_agent_list[i]:
                    data[i][agent_name] = int(current[i][agent_list.index(agent_name)])

            requests.post(report_url, json=data)
            print("A: ", data[0])
            print("B: ", data[1])
        else:
            print(current[0])
            print(current[1])

    for i, debugUi in enumerate(debug_ui_list):
        cv2.imshow("debug-" + str(i), debugUi)

    key = cv2.waitKey(1)

    # print(key)

    if (debug):
        if(key == 13):
            cfg["threshold0"]["white"].value = white_threshold[0]
            cfg["threshold1"]["white"].value = white_threshold[1]
            cfg.update_file()
        if(key == 49):
            white_threshold[0] -= 1
        if(key == 51):
            white_threshold[0] += 1
        if(key == 55):
            white_threshold[1] -= 1
        if(key == 57):
            white_threshold[1] += 1

        if(key == 27):
            cv2.destroyAllWindows()
            exit()

    endAt = time()
    # print("time: ", (endAt - startAt) * 1000, "ms")
    if endAt - startAt < oneFrameTime and not(debug):
        sleep(oneFrameTime - (endAt - startAt))
