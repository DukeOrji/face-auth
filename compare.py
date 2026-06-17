#compare.py
import faiss
import numpy as np
from get_user_face import get_user_emb

threshold = 0.2

def init_faiss(stored_face):
        #initialize faiss search
        if len(stored_face) == 0:
                raise ValueError("No approved face embeddings found")
        embedding_matrix = np.array(stored_face).astype("float32")
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)
        return index

#run faiss comparison
def comparison(index, stranger_vector):
        distance, indices = index.search(
                stranger_vector,
                k=1
        )

        distance = distance[0][0]
        if distance < threshold:
                return "Authorized", distance
        else:
                return "Unauthorized", distance
                