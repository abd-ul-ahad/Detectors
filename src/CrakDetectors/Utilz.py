import time
import cv2


class Utilz:
    starting_time = 0


    @staticmethod
    def show_fps(image, origin=(100, 100), font=cv2.FONT_HERSHEY_PLAIN, fontscale=3, color=(255, 0, 255), thickness=3 ):
        """
        image: image or numpy array
        origin: x, y coordinates where you want to show fps
        font: font style
        fontscale: font size
        color: rgb color code
        thickness: thickness of font
        """
        cTime = time.time()
        fps = 1/(cTime - Utilz.starting_time)
        Utilz.starting_time = cTime
        cv2.putText(image, str(int(fps)), origin,
                    font, fontscale, color, thickness)