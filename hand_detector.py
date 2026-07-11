import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, max_hands = 1, min_detection_confidence = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(max_num_hands = max_hands, min_detection_confidence = min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_features = None
        finger_x = 0
        finger_y = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                hand_features = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    hand_features.append(lm.x)
                    hand_features.append(lm.y)

                    if idx == 8:
                        finger_x = lm.x
                        finger_y = lm.y
                break

        return hand_features, finger_x, finger_y


