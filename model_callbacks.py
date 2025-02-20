import requests
import toml

with open('secrets.toml', 'r') as f:
    secrets = toml.load(f)

VGG_API_URL = secrets["urls"]["VGG_API"]
RESNET_API_URL = secrets["urls"]["RESNET_API"]
EFFICIENTNET_API_URL = secrets["urls"]["EFFICIENTNET_API"]
PHI_API_URL = secrets["urls"]["PALI_API"]
PALI_API_URL = secrets["urls"]["PHI_API"]
LLAMA_API_URL = secrets["urls"]["LLAMA_API"]

def vgg(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(VGG_API_URL, files={"file": image_file})
    return response.json()

def resnet(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(RESNET_API_URL, files={"file": image_file})
    return response.json()

def efficientnet(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(EFFICIENTNET_API_URL, files={"file": image_file})
    return response.json()

def phi_output(image_path, text_input):
    with open(image_path, "rb") as image_file:
        response = requests.post(PHI_API_URL, files={"file": image_file}, data={"text": text_input})
    return response.json()

def pali_output(image_path, text_input):
    with open(image_path, "rb") as image_file:
        response = requests.post(PALI_API_URL, files={"file": image_file}, data={"text": text_input})
    return response.json()

def llama_output(image_path, text_input):
    with open(image_path, "rb") as image_file:
        response = requests.post(LLAMA_API_URL, files={"file": image_file}, data={"text": text_input})
    return response.json()
