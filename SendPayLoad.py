from flask import Flask, jsonify, request
from face_utils import FaceUtils
import requests

face_utils = FaceUtils()

# Defining the paths to the model and the weight files
model_path = "antispoofing_models/antispoofing_model.json"
weight_path = "antispoofing_models/model1.h5"

app = Flask(__name__)


@app.route('/sendPayLoad', methods=['POST'])
def compare_faces():
    # Get the base64-encoded images from the POST request
    inputs = request.json
    inputFaceBase64 = inputs['input_image']
    testFaceBase64 = inputs['test_image']
    # ip = request.json['ip']
    # other = request.json['other']
    # isSpoof = request.json['spoof']

    # Initializing the output strings
    matchText = "No Face"
    spoofText = ""

    # Getting the images as np arrays from Base64
    inputFace = face_utils.base64_to_numpy(inputFaceBase64)
    testFace = face_utils.base64_to_numpy(testFaceBase64)

    # Cropping the images to only include the face
    croppedInput = face_utils.cropped_faces(inputFace)
    croppedTest = face_utils.cropped_faces(testFace)

    # Comparing the difference between the two faces
    difference = face_utils.compare_faces(inputFace, testFace)
    if difference:
        percentage = "{:.2f}%".format(max(0, min(100, 100 * (1 - (difference - 0.4) / (1 - 0.4)))))
        matchText = f"{percentage} Face Match"

        spoof = "{:.2f}%".format(face_utils.spoof_check(croppedTest, model_path, weight_path))
        spoofText = f"{spoof} Spoofing prediction"

    # Return the results as a JSON object
    second_api_endpoint = "http://localhost:5002/results"
    inputs['match_result'] = matchText
    inputs['spoof_result'] = spoofText
    # headers = {'Content-Type': 'application/json'}
    response = requests.post(second_api_endpoint, json=inputs)

    # Check the response from the second API endpoint
    if response.status_code == 200:
        return jsonify({'message': 'Results sent to second API endpoint'})
    else:
        return jsonify({'error': f'Request failed with status code {response.status_code}'})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5001)
