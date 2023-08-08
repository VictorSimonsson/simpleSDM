import json


fp = "array12.json"

with open(fp) as file:
    microphones = json.load(file)


print(microphones['MicrophoneLayout']["Microphones"][0])