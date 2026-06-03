"""
this program captures the facial structures of a user
saves its dimensions 
then compares those dimensions against
that of a stranger appearing on the camera
"""

import cv2
import face_recognition as fr
import numpy as np
import time

stored_face = []

def get_user_fc():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    for i in range(4):
        ret, frame = cap.read()
        cv2.imshow("FirstW", frame)

        cv2.imwrite(f'{i}new_img.jpg', frame)
        time.sleep(4) #wait for more motion to improve accuracy

        key = cv2.waitKey(1)
        if key == ord("p"):
            break

    for x in range(4):
        img = fr.load_image_file(f'{x}new_img.jpg')
        
        stored_face_encoding = fr.face_encodings(img)
        if len(stored_face_encoding) > 0:
            stored_face.append(stored_face_encoding)    

    cap.release()
    cv2.destroyAllWindows()


get_user_fc()

