from huggingface_hub import login, whoami
import toml

with open('secrets.toml', 'r') as f:
    secrets = toml.load(f)

token = secrets["huggingface_creds"]["hf_token"]

def login_into_hf():
    login(token=token)

    try:
        user_info = whoami()
        print("Login successful!")
        print("User info:", user_info)
    except Exception as e:
        print("Login failed:", e)

