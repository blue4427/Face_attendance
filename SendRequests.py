import requests

# Define the API endpoint URL
url = 'http://localhost:5001/sendPayLoad'

# Define the input and test image base64 strings
with open('inputFace.txt', 'r') as f:
    input_base64 = f.read()

with open('testFace.txt', 'r') as f:
    test_base64 = f.read()

# Define the payload data
data = {
    'input_image': input_base64,
    'test_image': test_base64,
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
