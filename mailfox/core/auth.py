import os

CREDENTIALS_FILE = os.path.expanduser("~/.mailfox/credentials")

def save_credentials(username: str, password: str, api_key: str = None):
    os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
    with open(CREDENTIALS_FILE, "w") as f:
        f.write(f"username={username}\n")
        f.write(f"password={password}\n")
        if api_key:
            f.write(f"api_key={api_key}\n")

def read_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError("Credentials file not found.")
    with open(CREDENTIALS_FILE, "r") as f:
        lines = f.readlines()
        username = lines[0].split("=")[1].strip()
        password = lines[1].split("=")[1].strip()
        api_key = lines[2].split("=")[1].strip() if len(lines) > 2 else None
    return username, password, api_key

def remove_credentials_file():
    if os.path.exists(CREDENTIALS_FILE):
        os.remove(CREDENTIALS_FILE)