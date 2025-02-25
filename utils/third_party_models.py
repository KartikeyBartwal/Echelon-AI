import requests
from collections import deque
from dotenv import load_dotenv 
import os 

load_dotenv()

YOUR_DEEPAI_API_KEY = os.getenv("YOUR_DEEPAI_API_KEY")
class APIModelQueue:
    def __init__(self, api_endpoints):
        """
        Initialize the queue with API endpoints.
        :param api_endpoints: List of API endpoint configurations (dicts with 'url' and 'headers' keys)
        """
        self.api_queue = deque(api_endpoints)

    def generate_response(self, prompt):
        """
        Attempts to get a response from available API endpoints.
        :param prompt: The user query to send to the API.
        :return: The first successful response or None if all fail.
        """
        while self.api_queue:
            api = self.api_queue.popleft()
            try:
                response = requests.post(api['url'], json={"prompt": prompt}, headers=api.get('headers', {}), timeout=10)
                if response.status_code == 200:
                    return response.json().get("response", response.text)  # Adjust based on API response format
            except requests.RequestException as e:
                print(f"Request failed: {e}") 
        return None  # Return None if all APIs fail

# Example API configurations (Replace with actual endpoints and authentication headers)
api_endpoints = [
    {"url": "https://api.deepai.org/api/text-generator", "headers": {"api-key": "YOUR_DEEPAI_API_KEY"}},
]

api_model_queue = APIModelQueue(api_endpoints)

# Example Usage
response = api_model_queue.generate_response("Hello, how are you?")
print(response)