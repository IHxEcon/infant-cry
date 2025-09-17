import requests

# Change this to your Render URL after deployment
API_URL = "http://127.0.0.1:5000/predict"   # local Flask
# API_URL = "https://your-app-name.onrender.com/predict"  # render URL

# Path to an audio file for testing
audio_file = "data/cry/1c.wav"

# Send file to API
with open(audio_file, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

# Print the API response
print("Status Code:", response.status_code)
print("Response:", response.json())
