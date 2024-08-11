import cv2
import numpy as np

camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

result = np.zeros((10, 10), np.uint8)

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

while True:
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

    print(median)

    # y = 85

    # for i in range(0, 5):
    #     cv2.line(extracted_frame, (446 + i * 66, y), (446 + round(40 * median[i]) + i * 66, y), (255, 255, 0), 2, cv2.LINE_4)
    #     cv2.putText(extracted_frame, str(round(median[i] * 100)), (446 + i * 66 + 4, y - 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))

    # for i in range(0, 5):
    #     cv2.line(extracted_frame, (1171 + i * 66, y), (1171 + round(40 * median[i + 5]) + i * 66, y), (255, 255, 0), 2, cv2.LINE_4)
    #     cv2.putText(extracted_frame, str(round(median[i + 5] * 100)), (1171 + i * 66 + 4, y - 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
 
    # cv2.imshow('camera', extracted_frame)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
