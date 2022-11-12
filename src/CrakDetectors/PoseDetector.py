import cv2
import numpy as np
import mediapipe as mp


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
                    result.append([id, cx, cy])

        return {"image": image, "landmarks": result}