import cv2
import numpy as np
from skimage.feature import hog
import joblib

capture = cv2.VideoCapture(0)

model = joblib.load('svm_hand_model.pkl')

training_data = []
label = []

while True:
    ret, frame = capture.read()


    roi = frame[0:300, 0:300]

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_limit = np.array([0, 30, 80], dtype = 'uint8')
    upper_limit = np.array([18, 255, 255], dtype = 'uint8')

    thresh = cv2.inRange(roi_hsv, lower_limit, upper_limit)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
    cv2.imshow('HSV_image', roi_hsv)
    cv2.imshow('HOG_visualization', hog_image_rescaled)
    cv2.imshow('THRESH_image', thresh)


    key = cv2.waitKey(1) & 0xFF

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


capture.release()

x = np.array(training_data)
y = np.array(label)

np.save('x_data.npy', x)
np.save('y_data.npy', y)
print('Data has been saved.')

cv2.destroyAllWindows()



