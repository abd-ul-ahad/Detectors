from .HandDetector import HandDetector
from .FaceDetector import FaceDetector
from .PoseDetector import PoseDetector
from .FaceMesh import FaceMesh
import math


def distance_formula(x1, y1, x2, y2):
    x = (x2-x1)*(x2-x1)
    y = (y2-y1)*(y2-y1)
    d = math.sqrt(x+y)
    return d