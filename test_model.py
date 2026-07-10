import cv2
import numpy as np
import mediapipe as mp
import joblib

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

    label = 'no_hand'

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_features = []
            for lm in hand_landmarks.landmark:
                hand_features.append(lm.x)
                hand_features.append(lm.y)

            input_data = np.array([hand_features])
            prediction = model.predict(input_data)
            label = prediction[0]

    if label == 'opened':
        cv2.putText(frame, 'Hand opened', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    elif label == 'closed':
        cv2.putText(frame, 'Hand closed', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    elif label == 'nothing':
        cv2.putText(frame, 'Wrong gesture', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No hand', (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('TEST MODELU SVM', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



