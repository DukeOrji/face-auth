#system.py
import cv2
import face_recognition as fr
import numpy as np
from compare import comparison, init_faiss
from get_user_face import get_user_fc, get_user_emb, img_path
import os


def visual_result(stranger_frame, results, distance, left, top):
    if results == "Authorized":
        color_code = (0, 255, 0)
    else:
        color_code = (0, 0, 255)

    cv2.putText(
        stranger_frame,
        f"{results}: {distance:.2f}",
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_code,
        2
    )

known_img_path =  img_path
if len(os.listdir(known_img_path)) == 0: #check if program has known images
    get_user_fc()
    stored_face = get_user_emb()

else:
    stored_face = get_user_emb()

index = init_faiss(stored_face)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_face_location = None
last_results = None
last_distance = None

frame_counter = 0
while True:
    
    ret, stranger_frame = cap.read()
    if not ret:
        print("didn't capture stranger frame")
        break
    frame_counter += 1

    if frame_counter % 3 == 0:        
        face_locations = fr.face_locations(stranger_frame)
        stranger_encoding = fr.face_encodings(stranger_frame, face_locations)

        if len(stranger_encoding) > 0:
            stranger_emb = stranger_encoding[0]
            stranger_vector = stranger_emb.astype("float32").reshape(1, -1)

            # print(embedding_matrix.shape)
            # print(stranger_vector.shape)

            results, distance = comparison(index, stranger_vector)
            if len(stranger_encoding) == 0:
                last_face_location = None
            last_face_location = face_locations[0]
            last_results = results
            last_distance = distance

    if last_face_location is not None:
        top, right, bottom, left = last_face_location

        if last_results == "Authorized":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            stranger_frame,
            (left, top),
            (right, bottom),
            color,
            2
        ) 

        visual_result(stranger_frame, last_results, last_distance, left, top)       

    cv2.imshow("SecondW", stranger_frame)

    key = cv2.waitKey(1)
    if key == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()