import cv2
import mediapipe as mp
import numpy as np


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
                        cx, cy = int(h*landmark.x), int(w*landmark.y)
                        result.append([id, cx, cy, z])

        return result



class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.pose_file = mp.solutions.pose

        self.pose_class = self.pose_file.Pose()

        self._draw_ = mp.solutions.drawing_utils

    def find_pose(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: print landmarks on the given image
        """
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.pose_class.process(imgRGB)

        if draw_landmarks:
            if processed.pose_landmarks:
                # for hand_landmarks in processed.multi_hand_landmarks:
                self._draw_.draw_landmarks(
                    image, processed.pose_landmarks, self.pose_file.POSE_CONNECTIONS)

        return image

    def get_pose_landmarks(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: return array of landmarks consist of id x-axis, y-axis, z-axis
        """
        result = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.pose_class.process(imgRGB)

        if draw_landmarks is True:
            if processed.pose_landmarks:
                # for pose_landmark in processed.pose_landmarks:
                self._draw_.draw_landmarks(
                    image, processed.pose_landmarks, self.pose_file.POSE_CONNECTIONS)
                for id, landmark in enumerate(processed.pose_landmarks.landmark):
                    h, w, z = image.shape
                    cx, cy = int(w*landmark.x), int(h*landmark.y)
                    result.append([id, cx, cy, z])

        return result

