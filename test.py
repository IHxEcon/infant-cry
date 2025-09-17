import requests

# Replace this with your ngrok public URL
url = "https://3e9d93b935c2.ngrok-free.app/predict"

# Replace 'test_audio.wav' with the path to your audio file
files = {"file": open("data/not_cry/27n.ogg", "rb")}

response = requests.post(url, files=files)
print(response.json())
