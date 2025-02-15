from openai import OpenAI
import requests
def payload(api_key, prompt, image, model):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Instead of combining the prompt with the full image data, send only the prompt.
    # Alternatively, if you really need to include an image reference, you can provide a short URL.
    payload_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload_data)
    response_json = response.json()
    
    if "choices" not in response_json or not response_json["choices"]:
        print("Error in GPT API response:")
        print(response_json)
        raise ValueError("GPT API response does not contain 'choices'.")
    
    message_content = response_json['choices'][0]['message']['content']
    return message_content

