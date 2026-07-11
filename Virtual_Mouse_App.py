import numpy as np
import joblib
import cv2
from hand_detector import HandDetector
from smooth_mouse import SmoothMouse

class VirtualMouseApp:
    def __init__(self, model_path='svm_hand_model.pkl'):
        self.model = joblib.load('svm_hand_model.pkl')
        print('SVM model loaded.')

        self.detector = HandDetector()
        self.mouse = SmoothMouse()

        self.capture = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.capture.read()

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            hand_features, fx, fy = self.detector.process_frame(frame)

            label = 'No hand'

            if hand_features is not None:
                input_data = np.array([hand_features])
                prediction = self.model.predict(input_data)
                label = prediction[0]

            if label == 'opened':
                cv2.putText(frame, 'Hand opened - Moving', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                self.mouse.move(fx, fy)

            if label == 'closed':
                cv2.putText(frame, 'Hand closed - clicking', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3,cv2.LINE_AA)
                self.mouse.click()

            cv2.imshow('Virtual Mouse', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()










