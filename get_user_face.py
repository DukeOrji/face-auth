#get user face.py
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
import os

img_path = r"C:\Users\duken\OneDrive\Desktop\face_auth\saved_face"
save_dir = "saved_face"
os.makedirs(save_dir, exist_ok=True)

def get_user_fc():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    for i in range(5):
        ret, frame = cap.read()
        cv2.imshow("Enrolling Facial Identification...", frame)

        time.sleep(2)#permit more motion to imporve accuracy
        file_path = os.path.join(save_dir, f"{i+1}new_img.jpg")
        cv2.imwrite(file_path, frame)

        key = cv2.waitKey(1)
        if key == ord("p"):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_user_emb():
    stored_face = []
    for filename in os.listdir(img_path):
            known_img = fr.load_image_file(os.path.join(img_path, filename))
            known_img_emb = fr.face_encodings(known_img)
            if len(known_img_emb) > 0:
                stored_face.append(known_img_emb[0])
    
    return stored_face

