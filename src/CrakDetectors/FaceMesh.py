import cv2
import numpy as np
import mediapipe as mp


class FaceMesh:
    def __init__(self, static_image_mode=False, max_nums_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode, max_nums_faces, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def get_face_mesh_landmarks(self, image, draw_landmarks=True):
        faces = []

        rgbFrame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed = self.FaceMesh.process(rgbFrame)

        if processed.multi_face_landmarks:
            for faceLandmarks in processed.multi_face_landmarks:
                if draw_landmarks is True:
                    self.mpDraw.draw_landmarks(
                        image, faceLandmarks, self.mpFaceMesh.FACEMESH_TESSELATION)

                face = []
                for landmark in faceLandmarks.landmark:
                    h, w, z = image.shape
                    cx, cy = int(h*landmark.x), int(w*landmark.y)
                    face.append([cx, cy])
                faces.append(face)

        return {"image": image, "faces": faces}
