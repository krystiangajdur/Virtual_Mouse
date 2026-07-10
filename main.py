import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)
training_data = []
label = []

while True:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_features = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_features = []
            for lm in hand_landmarks.landmark:
                hand_features.append(lm.x)
                hand_features.append(lm.y)

            current_features = hand_features

    cv2.imshow('Virtual_Mouse-Test', frame)

    key = cv2.waitKey(50)

    if key == ord('q'):
        break
    elif key == ord('z') and current_features is not None:
        training_data.append(current_features)
        label.append('opened')
        print('OPENED added, total:', len(label))
    elif key == ord('x') and current_features is not None:
        training_data.append(current_features)
        label.append('closed')
        print('CLOSED added, total:', len(label))
    elif key == ord('c') and current_features is not None:
        training_data.append(current_features)
        label.append('nothing')
        print('NOTHING added, total:', len(label))


capture.release()
cv2.destroyAllWindows()


x = np.array(training_data)
y = np.array(label)
np.save('x_data.npy', x)
np.save('y_data.npy', y)
print('Data saved successfully!')


