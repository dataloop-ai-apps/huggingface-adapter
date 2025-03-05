import openai
import os
import time
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CARD_TYPE = 'url'
MODEL_REPO = 'nlpconnect/vit-gpt2-image-captioning'

# Retrieve the API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

usage_file = os.path.join(os.getcwd(), 'usage.txt')


# don't use gpt4, too $$$
def get_completion(prompt, model="gpt-4o-mini"):  # gpt-4o-mini-2024-07-18
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0  # automatically increase temp until certain thresholds are hit
    )
    return response.choices[0].message.content, response.usage


def script_to_string(file_path):
    """Reads a Python script and converts it into a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}"


# Example usage
script_path = "example_script.py"  # Replace with your script's path
script_string = script_to_string(script_path)
print(script_string)


# get the model card to be generated
if CARD_TYPE == 'file':
    model_card_path = hf_hub_download(repo_id=MODEL_REPO, filename='README.md')
    with open(model_card_path, 'r') as f:
        model_card = f.read()
else:
    model_card = rf"https://huggingface.co/{MODEL_REPO}"

# give an example of a model card -> adapter
ex1_repo = 'Salesforce/blip2-opt-2.7b'
ex1_adapter_url = 'https://github.com/dataloop-ai-apps/huggingface-adapter/blob/APPS-1558-add-BLIP-2-model-adapter/adapters/blip_2/blip_2.py'

ex2_repo = 'Salesforce/blip-image-captioning-base'
ex2_adapter_url = 'https://github.com/dataloop-ai-apps/huggingface-adapter/blob/APPS-1590-refactor-BLIP-model-adapter/adapters/blip_image_captioning_large/blip_image_captioning_large.py'

# fxn_to_include = ['load', 'prepare_item_func', 'predict', 'train', 'reformat_messages', 'compute_confidences']
fxns_to_incl = ['load', 'prepare_item_func', 'predict']

system_prompt = rf"""
Your task is to generate code that creates a new Python class called \
`HuggingAdapter`. This class should inherit from the `BaseModelAdapter` class in the dtlpy library. \
(link: https://github.com/dataloop-ai/dtlpy/blob/master/dtlpy/ml/base_model_adapter.py). \
Use the example below to complete the task. \n\n

Model adapter example:\
 1: This is an adapter for the {ex1_repo} model created based on the model info here: {ex1_adapter_url}. \n \\
 2: This is an adapter for the {ex2_repo} model created based on the model info here: {ex2_adapter_url}. \n \\

The `HuggingAdapter` class should have the following methods: {fxns_to_incl}. \n 
Using the model info provided below, use the examples to generate the HuggingAdapter class script for the new \
model. Pay special attention to the code snippets in the model card, and include the relevant code in the "predict"\
 or "prepare_item_func" functions of the HuggingAdapter off of it. No init function should be included, as the class\
 is initialized with the `BaseModelAdapter` class. The load function should include loading configurations from \
 `self.configuration.get`
 
 
Don't repeat this prompt and keep the response as concise as possible.\

Model Info: ```{model_card}```\
"""

response, usage = get_completion(system_prompt)
response = response.replace(r"```python", "")
response = response.replace(r"```", "")

with open(
    os.path.join(os.getcwd(), "responses", f"{MODEL_REPO.split('/')[1]}_{time.strftime('%Y.%m.%d_%H.%M.%S')}.py"), "w"
) as f:
    f.write(response)

with open(usage_file, 'a') as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- {MODEL_REPO}: {usage}\n")

print("done!")
