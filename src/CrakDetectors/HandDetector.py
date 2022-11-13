import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode_=False, maximum_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.mode = static_image_mode_
        self.maximum_num_hands = maximum_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.hand_file = mp.solutions.hands
        self.hand_class = self.hand_file.Hands()
        self._draw_ = mp.solutions.drawing_utils

    def find_hands(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: print landmarks on the given image
        """
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.hand_class.process(imgRGB)

        if draw_landmarks is True:
            if processed.multi_hand_landmarks:
                for hand_landmarks in processed.multi_hand_landmarks:
                    self._draw_.draw_landmarks(
                        image, hand_landmarks, self.hand_file.HAND_CONNECTIONS)

        return image

    def get_hand_landmarks(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: return array of landmarks consist of id x-axis, y-axis, z-axis
        """
        result = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.hand_class.process(imgRGB)

        if draw_landmarks is True:
            if processed.multi_hand_landmarks:
                for hand_landmarks in processed.multi_hand_landmarks:
                    self._draw_.draw_landmarks(
                        image, hand_landmarks, self.hand_file.HAND_CONNECTIONS)
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        h, w, z = image.shape
                        cx, cy = int(w*landmark.x), int(h*landmark.y)
                        result.append([id, cx, cy])

        return {"image": image, "landmarks": result}