import cv2
import face_recognition as fr
import numpy as np
from compare import comparison, embedding_matrix
from get_user_face import get_user_fc

def visual_result(stranger_frame, results, distance, left, top):
    if result == "Authorized":
        color_code = (0, 255, 0)
    else:
        color_code = (0, 0, 255)

        cv2.putText(
            stranger_frame,
            (left, top - 10),
            f"{result}: {distance:.2f}",
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_code,
            2
        )



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, stranger_frame = cap.read()

    if not ret:
        print("didn't capture stranger frame")
    
    face_locations = fr.face_locations(stranger_frame)
    stranger_encoding = fr.face_encodings(stranger_frame, face_locations)

    if len(stranger_encoding) > 0:
        stranger_emb = stranger_encoding[0]
        stranger_vector = stranger_emb.astype("float32").reshape(1, -1)

        print(embedding_matrix.shape)
        print(stranger_vector.shape)
        
        result, distance = comparison(stranger_vector)

        top, right, bottom, left = face_locations[0]

        #draw bounadry lines
        if result == "Authorized":
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

        visual_result(stranger_frame, result, distance, left, top)

    cv2.imshow("SecondW", stranger_frame)
    key = cv2.waitKey(1)
    if key == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()