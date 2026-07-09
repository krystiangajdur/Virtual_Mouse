import cv2
import numpy as np
from skimage.feature import hog

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    roi = frame[0:300, 0:300]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    fd, hog_image = hog(
        roi_gray,
        orientations = 9,
        pixels_per_cell = (8,8),
        cells_per_block = (2,2),
        visualize = True
    )

    hog_image_rescaled = np.uint8(hog_image * 255)

    cv2.rectangle(frame, (0,0), (300, 300), (0, 255, 0), 3)

    cv2.imshow('Image', frame)
    cv2.imshow('Gray_hand', roi_gray)
    cv2.imshow('HOG_visualization', hog_image_rescaled)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()



