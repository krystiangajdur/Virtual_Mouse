import pyautogui
import cv2
import numpy as np
import mediapipe as mp
import joblib

pyautogui.FAILSAFE = True
screen_w, screen_h = pyautogui.size()

model = joblib.load('svm_hand_model.pkl')
print('SVM Model loaded.')

mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_features = None
    finger_x = 0
    finger_y = 0

    label = 'no_hand'

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_features = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                hand_features.append(lm.x)
                hand_features.append(lm.y)

                if idx == 8:
                    finger_x = lm.x
                    finger_y = lm.y

            input_data = np.array([hand_features])
            prediction = model.predict(input_data)
            label = prediction[0]

            margin = 0.2
            mouse_x = int(np.interp(finger_x, [margin, 1.0 - margin], [0, screen_w]))
            mouse_y = int(np.interp(finger_y, [margin, 1.0 - margin], [0, screen_h]))

    if label == 'opened':
        cv2.putText(frame, 'Hand opened - moving', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        pyautogui.moveTo(mouse_x, mouse_y)
    elif label == 'closed':
        cv2.putText(frame, 'Hand closed - clicking', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        pyautogui.click()
        cv2.waitKey(200)
    cv2.imshow('TEST MODELU SVM', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()





