import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_face_detection_conf=0.5) -> None:
        self.min_detect_conf = min_face_detection_conf

        self.face_file = mp.solutions.face_detection
        self.Face_class = self.face_file.FaceDetection(self.min_detect_conf)
        self._draw_ = mp.solutions.drawing_utils

    def find_face(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: print landmarks on the given image
        """
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.Face_class.process(imgRGB)

        if draw_landmarks is True:
            if processed.detections:
                for lms in processed.detections:
                    self._draw_.draw_detection(image, lms)

                    lm = lms.location_data.relative_bounding_box
                    h, w, c = image.shape

                    box = int(w*lm.xmin), int(h*lm.ymin), int(w *
                                                              lm.width), int(h*lm.height)

                    cv2.rectangle(image, box, (255, 0, 255), 2)
                    cv2.putText(
                        image, f"{int(lms.score[0]*100)}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        return image

    def get_face_landmarks(self, image, draw_landmarks=True):
        """
        Image: cv2 BGR Image,
        return: return array of landmarks consist of id x-axis, y-axis, z-axis
        """
        faces = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = self.Face_class.process(imgRGB)

        if processed.detections:
            for id, lms in enumerate(processed.detections):

                lm = lms.location_data.relative_bounding_box
                h, w, c = image.shape

                box = int(w*lm.xmin), int(h*lm.ymin), int(w *
                                                          lm.width), int(h*lm.height)

                if draw_landmarks:
                    self._draw_.draw_detection(image, lms)
                    cv2.rectangle(image, box, (255, 0, 255), 2)
                    cv2.putText(
                        image, f"{int(lms.score[0]*100)}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

                faces.append([id, box, lms.score])

        return {"image": image, "faces": faces}
