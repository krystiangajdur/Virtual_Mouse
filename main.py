import cv2
import numpy as np
from skimage.feature import hog
import joblib

capture = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

cv2.createTrackbar("L-H", "Calibration", 0, 179, nothing)   # Lower Hue
cv2.createTrackbar("L-S", "Calibration", 30, 255, nothing)  # Lower Saturation
cv2.createTrackbar("L-V", "Calibration", 80, 255, nothing)  # Lower Value

cv2.createTrackbar("U-H", "Calibration", 18, 179, nothing)  # Upper Hue
cv2.createTrackbar("U-S", "Calibration", 255, 255, nothing) # Upper Saturation
cv2.createTrackbar("U-V", "Calibration", 255, 255, nothing) # Upper Value

model = joblib.load('svm_hand_model.pkl')

training_data = []
label = []

while True:
    ret, frame = capture.read()
    roi = frame[0:300, 0:300]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    lh = cv2.getTrackbarPos("L-H", "Calibration") #Sets the minimum color shade
    ls = cv2.getTrackbarPos("L-S", "Calibration") #Sets the minimum color intensity
    lv = cv2.getTrackbarPos("L-V", "Calibration") #Sets the minimum brightness
                                                                      #Sets the same but the maximum level:
    uh = cv2.getTrackbarPos("U-H", "Calibration")
    us = cv2.getTrackbarPos("U-S", "Calibration")
    uv = cv2.getTrackbarPos("U-V", "Calibration")


    lower_limit = np.array([lh, ls, lv], dtype="uint8")
    upper_limit = np.array([uh, us, uv], dtype="uint8")

    thresh = cv2.inRange(roi_hsv, lower_limit, upper_limit) #thresh means white or black

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    fd, hog_image = hog(
        thresh,
        orientations = 9,
        pixels_per_cell = (8,8),
        cells_per_block = (2,2),
        visualize = True
    )

    hog_image_rescaled = np.uint8(hog_image * 255)

    prediction = model.predict([fd])

    gesture = prediction[0]
    cv2.putText(frame, f'{gesture}', (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame, (0,0), (300, 300), (0, 255, 0), 3)

    cv2.imshow('image', frame)
    #cv2.imshow('HSV_image', roi_hsv)
    #cv2.imshow('HOG_visualization', hog_image_rescaled)
    cv2.imshow('THRESH_image', thresh)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('z'):
        training_data.append(fd)
        label.append('opened')
        print('OPENED hand added, index:', len(label))
    elif key == ord('c'):
        training_data.append(fd)
        label.append('closed')
        print('CLOSED hand added, index: ', len(label))
    elif key == ord('x'):
        training_data.append(fd)
        label.append('nothing')
        print('NOTHING added, index:', len(label))


capture.release()

x = np.array(training_data)
y = np.array(label)

np.save('x_data.npy', x)
np.save('y_data.npy', y)
print('Data has been saved.')

cv2.destroyAllWindows()




