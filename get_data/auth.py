import requests
from dotenv import load_dotenv
import os
from get_data.models import LoginResponse

load_dotenv()


def login() -> LoginResponse:
    """Login the user with enviroment set email and password.

    Returns:
        dict: User object containing user details.
    Raises:
        Exception: If the login fails.
    """
    url = "https://api.kickbase.com/v4/user/login"
    # JSON payload for the request
    # user needs to add email and password

    payload = {
        "em": os.getenv("EMAIL"),
        "loy": False,
        "pass": os.getenv("PASSWORD"),
        "rep": {},
    }

    # headers for the request
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # sending the POST request
    try:
        response = requests.post(url, json=payload, headers=headers)
        # Extracting the token from the response JSON
        response.raise_for_status()
        user = LoginResponse(**response.json())
    except Exception as e:
        raise Exception(f"Login failed: {e}")
    return user

