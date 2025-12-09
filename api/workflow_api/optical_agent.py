import requests
import os
import uuid
from dotenv import load_dotenv


# Load secrets from .env (only for local development)
load_dotenv()  # Loads variables from .env into OS environment

# Read from environment variables (fallback to placeholders for public code)
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY", "YOUR_LANGFLOW_API_KEY_HERE")
LANGFLOW_HOST_URL = os.getenv("LANGFLOW_HOST_URL", "http://localhost:7860")
RESEARCH_FLOW_ID = os.getenv("RESEARCH_FLOW_ID", "YOUR_RESEARCH_FLOW_ID_HERE")

# Build API URL dynamically
url = f"{LANGFLOW_HOST_URL}/api/v1/run/{RESEARCH_FLOW_ID}"

# Request payload configuration
payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "topological photonics",
    "session_id": str(uuid.uuid4())
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": LANGFLOW_API_KEY
}


try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")