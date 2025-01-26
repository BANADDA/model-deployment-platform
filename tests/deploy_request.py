import requests
import json

# Define the endpoint URL
url = "http://localhost:8000/v1/deployments/27f85e8a-a0a7-45e8-b055-dddd5f5f1f39/predict"

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Define the payload
payload = {
    "prompt": "Write me a for loop to print 5 numbers in python",
    "max_tokens": 100
}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()

    # Pretty-print the entire JSON response (optional)
    print("Full JSON Response:")
    print(json.dumps(data, indent=4))
    print("\n" + "="*50 + "\n")

    # Extract the 'response' field
    generated_response = data.get("response", "")

    # Print the generated response
    print("Generated Response:")
    print(generated_response)

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")  # HTTP error
except Exception as err:
    print(f"An error occurred: {err}")          # Other errors
