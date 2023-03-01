import face_recognition
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Getting rid of the tensorflow messages upon initialization
from tensorflow.keras.models import model_from_json


class FaceUtils:
    def __init__(self):
        pass

    def cropped_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return frame
        top, right, bottom, left = face_locations[0]
        tolerance = 5
        top -= tolerance
        bottom += tolerance
        left -= tolerance
        right += tolerance
        face = frame[top:bottom, left:right]
        return face

    def base64_to_numpy(self, base64_str):
        im = Image.open(BytesIO(base64.b64decode(base64_str)))
        return np.array(im)

    def compare_faces(self, face1, face2):
        # Load the two faces into numpy arrays
        face1_encoding = face_recognition.face_encodings(face1, model="large")[0]
        face2_encoding = face_recognition.face_encodings(face2, model="large")
        for faces in face2_encoding:
            distance = np.linalg.norm(face1_encoding - faces)
            return distance

    def spoof_check(self, img, model_path, weight_path):

        # Load the anti-spoofing model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weight_path)
        img = Image.fromarray(img)
        resized_face = img.resize((160, 160))
        resized_face = np.array(resized_face) / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)
        predictions = model.predict(resized_face)[0]
        percent = predictions.item() * 100
        return percent
