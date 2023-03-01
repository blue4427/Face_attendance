from face_utils import FaceUtils
import argparse
import matplotlib.pyplot as plt

face_utils = FaceUtils()

# Defining the paths to the model and the weight files
model_path = "antispoofing_models/antispoofing_model.json"
weight_path = "antispoofing_models/model1.h5"

if __name__ == '__main__':
    # Setting the arguments
    parser = argparse.ArgumentParser(description='Compare faces and check for spoofing.')
    parser.add_argument('input_file', type=str, help='Input file path')
    parser.add_argument('test_file', type=str, help='Test file path')
    parser.add_argument('--spoof', action='store_true', help='Check for spoofing')
    args = parser.parse_args()

    # Reading the Base64 Image from the text file provided by the args
    with open(args.input_file, 'r') as f:
        inputFaceBase64 = f.read()

    with open(args.test_file, 'r') as f:
        testFaceBase64 = f.read()

    # Initializing the output strings
    matchText = "Face not found"
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
        matchText = f"Face Match: {percentage}"

        # Checking for spoofing only if the spoofing argument is passed
        if args.spoof:
            spoof = "{:.2f}%".format(face_utils.spoof_check(croppedTest, model_path, weight_path))
            spoofText = f"Spoofing Prediction: {spoof}"
            print(spoofText)

    print(matchText)

    # Displaying the results as a plt diagram
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(croppedInput)
    ax1.set_title('Input Image')
    ax2.imshow(croppedTest)
    ax2.set_title('Test Image')
    plt.figtext(0.5, 0.12, matchText, ha='center', size='large')
    plt.figtext(0.5, 0.05, spoofText, ha='center', size='large')
    plt.show()
