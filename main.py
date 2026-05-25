import cv2
import face_recognition as fr
import faiss
import numpy as np
import os

img_path = r"C:\Users\duken\OneDrive\Desktop\face_auth\approved_face"
approved_face_emb = []
threshold = 0.2
color_code = (0, 0, 0)

def visual_result(frame, result, distance, left, top):

    if result == "Authorized":
        color_code = (0, 255, 0)

    else:
        color_code = (0, 0, 255)

    cv2.putText(
        frame,
        f"{result}: {distance:.2f}",
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_code,
        2
    )


def comparison(stranger_vector):
    distance, indices = index.search(
        stranger_vector,
        k=2
    )

    distance = distance[0][0]

    if distance < threshold:
        return "Authorized", distance
    else:
        return "Unauthorized", distance

#image embedding
for filename in os.listdir(img_path):
    file_path = os.path.join(img_path, filename)
    known_img = fr.load_image_file(file_path)
    known_img_emb = fr.face_encodings(known_img)
    if len(known_img_emb) > 0:
        approved_face_emb.append(known_img_emb[0])

#initialize faiss
if len(approved_face_emb) == 0:
    raise ValueError("No approved face embeddings found.")
embedding_matrix = np.array(approved_face_emb).astype("float32")
dimension = embedding_matrix.shape[1] #128
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)    


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("didnt capture frame")
        break


    face_locations = fr.face_locations(frame)
    encodings = fr.face_encodings(frame, face_locations)

    if len(encodings) > 0:
        stranger_emb = encodings[0]
        stranger_vector = stranger_emb.astype("float32").reshape(1, -1)
        result, distance = comparison(stranger_vector)

        top, right, bottom, left = face_locations[0]

        #draw boundary lines
        if result == "Authorized":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        visual_result(frame, result, distance, left, top)

        cv2.imshow("camera", frame)
    key = cv2.waitKey(1)

    if key == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()

