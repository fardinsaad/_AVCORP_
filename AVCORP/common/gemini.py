import os
import backoff
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Load environment variables from the .env file
# print("Before loading .env:", os.getenv('GEMINI_API_KEY'))
load_dotenv(override=True)  # Force override existing environment variables
# print("After loading .env:", os.getenv('GEMINI_API_KEY'))

# Set Gemini API key from environment variable
key = os.getenv('GEMINI_API_KEY')

if not key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=key)

@backoff.on_exception(backoff.expo, errors.ClientError, max_time=120)
def get_gemini_completion(**kwargs):
    """
    Get completion from Gemini API with backoff in case of errors.
    
    Usage:
        response = get_gemini_completion(
            model="gemini-3.1-pro-preview",
            prompt="Your prompt here",
            max_output_tokens=512
        )
    """
    model_name = kwargs.get('model', 'gemini-3.1-pro-preview')
    prompt = kwargs.get('prompt', kwargs.get('messages', ''))
    max_output_tokens = kwargs.get('max_output_tokens', kwargs.get('max_tokens', 512))

    system_instruction = None
    contents = prompt

    # Handle messages format (OpenAI-style) for compatibility
    if isinstance(prompt, list):
        user_parts = []
        for msg in prompt:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                user_parts.append(content)
        contents = '\n'.join(user_parts)

    config_kwargs = {'max_output_tokens': max_output_tokens}
    if system_instruction:
        config_kwargs['system_instruction'] = system_instruction

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Return response in a format similar to OpenAI
    class GeminiResponse:
        def __init__(self, text):
            self.text = text.strip()
            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': self.text})()})]

    return GeminiResponse(response.text)

def get_response(prompt):
    """Function to interact with Gemini and generate a response."""
    response = client.models.generate_content(
        model='gemma-3-27b-it',
        contents=prompt,
        config={'max_output_tokens': 512}
    )
    return response.text.strip()

# Main function to run the script and test the API connection
if __name__ == '__main__':
    # Check if the API key is correctly loaded
    
    if key:
        print("API key loaded successfully.")
        # List all available Gemini models
        print("\nModels available for this API key:")
        try:
            models = client.models.list()
            for model in models:
                print(f"  {model.name}")
        except Exception as e:
            print(f"Error listing models: {e}")
            
        # Test with a sample prompt
        print("\nTesting API with sample prompt...")
        test_prompt = "Name three countries that start with K?"
        try:
            result = get_response(test_prompt)
            print(f"Gemini Response: {result}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("API key not found. Please set it in your .env file.")
