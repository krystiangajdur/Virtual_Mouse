import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    height = int(capture.get(4))
    width = int(capture.get(3))

    roi = frame[0:300, 0:300]

    cv2.imshow('hand', roi)

    cv2.imshow('full_image', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()

