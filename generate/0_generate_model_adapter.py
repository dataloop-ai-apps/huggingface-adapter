# This script generates a model adapter for a given model from Hugging Face.
# It uses existing, functional model adapters inherting from dl.BaseModelAdapter as examples to follow.
# It uses GPT-4o-mini to generate the adapter.

from os.path import exists
import openai
import os
import time
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from prompt_templates import ModelAdapterPrompts

# Load environment variables from .env file
load_dotenv()

MODEL_ADAPTER_TYPE = "model_adapter_prompt"
MODEL_REPO = 'google/pegasus-xsum'


# Setup OpenAI client
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


# load all relevant dtlpy context
dtlpy_context = script_to_string(os.path.join(os.getcwd(), 'dtlpy_context.py'))

# convert python scripts to strings for prompting
script_path = "adapters/blip_2/blip_2.py"  # Replace with your script's path
script_string1 = script_to_string(os.path.join(os.getcwd(), script_path))
# print(script_string1)

script_path = "adapters/blip_image_captioning_large/blip_image_captioning_large.py"  # Replace with your script's path
script_string2 = script_to_string(os.path.join(os.getcwd(), script_path))
# print(script_string2)

# original_adapter = script_to_string(os.path.join(os.getcwd(), original_adapter_path))

# get the model card to be generated
model_card_path = hf_hub_download(repo_id=MODEL_REPO, filename='README.md')
with open(model_card_path, 'r') as f:
    model_card = f.read()

# give an example of a model card -> adapter
ex1_repo = 'Salesforce/blip2-opt-2.7b'
ex2_repo = 'Salesforce/blip-image-captioning-base'

fxns_to_incl = ['load', 'prepare_item_func', 'predict']

# Get the template and render it with the required variables
template = ModelAdapterPrompts.get_template(MODEL_ADAPTER_TYPE)
user_prompt = template.render(
    fxns_to_incl=fxns_to_incl,
    dtlpy_context=dtlpy_context,
    ex1_repo=ex1_repo,
    script_string1=script_string1,
    ex2_repo=ex2_repo,
    script_string2=script_string2,
    model_card=model_card,
)

response, usage = get_completion(user_prompt)
response = response.replace(r"```python", "")
response = response.replace(r"```", "")

responses_dir = os.path.join(os.getcwd(), "generate", "responses")
os.makedirs(responses_dir, exist_ok=True)
responses_path = os.path.join(responses_dir, f"{MODEL_REPO.split('/')[1]}_{time.strftime('%Y.%m.%d_%H.%M.%S')}.py")
with open(responses_path, "w") as f:
    f.write(response)

with open(usage_file, 'a') as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- {MODEL_REPO}: {usage}\n")

print("done!")
