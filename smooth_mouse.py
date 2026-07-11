import numpy as np
import pyautogui
import cv2

class SmoothMouse:
    def __init__(self, smoothening = 3, margin = 0.2):
        pyautogui.FAILSAFE = True
        self.screen_w, self.screen_h = pyautogui.size()

        self.prev_x = 0
        self.prev_y = 0

        self.smoothening = smoothening
        self.margin = margin

    def move(self, finger_x, finger_y):
        mouse_x = int(np.interp(finger_x, [self.margin, 1.0 - self.margin], [0, self.screen_w]))
        mouse_y = int(np.interp(finger_y, [self.margin, 1.0 - self.margin], [0, self.screen_h]))

        cloc_x = self.prev_x + (mouse_x - self.prev_x) / self.smoothening
        cloc_y = self.prev_y + (mouse_y - self.prev_y) / self.smoothening

        pyautogui.moveTo(int(cloc_x), int(cloc_y))

        self.prev_x, self.prev_y = cloc_x, cloc_y

    def click(self):
        pyautogui.click()
        cv2.waitKey(200)


