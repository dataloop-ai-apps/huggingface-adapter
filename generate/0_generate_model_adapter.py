# This script generates a model adapter for a given model from Hugging Face.
# It uses existing, functional model adapters inherting from dl.BaseModelAdapter as examples to follow.
# It uses GPT-4o-mini to generate the adapter.

import os
import time
import openai
from huggingface_hub import hf_hub_download

from prompt_templates import ModelAdapterPrompts

# Set the template type to use for the development process, eventually it should only be "create_hugging_adapter"
MODEL_ADAPTER_TYPE = (
    "create_model_adapter"  # Options: "create_model_adapter", "create_hugging_adapter", "convert_to_hugging_base"
)
MODEL_REPO = 'ustc-community/dfine-xlarge-obj2coco'
if MODEL_ADAPTER_TYPE == "convert_to_hugging_base":
    ORIGINAL_ADAPTER_PATH = (
        "generate/responses/basemodeladapter/dfine-xlarge-coco_2025.06.04_15.10.00.py"  # Path to adapter to convert
    )
else:
    ORIGINAL_ADAPTER_PATH = None

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

script_path = (
    "generate/responses/basemodeladapter/detr-resnet-50_2025.03.19_12.30.00.py"  # Replace with your script's path
)
script_string2 = script_to_string(os.path.join(os.getcwd(), script_path))
# print(script_string2)

script_path = (
    "generate/responses/basemodeladapter/pegasus-xsum_2025.03.19_15.10.00.py"  # Replace with your script's path
)
script_string3 = script_to_string(os.path.join(os.getcwd(), script_path))
# print(script_string3)

if MODEL_ADAPTER_TYPE == "convert_to_hugging_base":
    original_adapter = script_to_string(os.path.join(os.getcwd(), ORIGINAL_ADAPTER_PATH))
    hugging_base_code = script_to_string(os.path.join(os.getcwd(), "generate/hugging_base.py"))
    # Example adapter using HuggingBase
    example_adapter = script_to_string(
        os.path.join(os.getcwd(), "generate/responses/detr-resnet-50_2025.03.19_14.44.08.py")
    )

# get the model card to be generated
model_card_path = hf_hub_download(repo_id=MODEL_REPO, filename='README.md')
with open(model_card_path, 'r') as f:
    model_card = f.read()

# Extract model type and media type from model card
model_type = "unknown"
media_type = "unknown"

# Try to determine model type from model card
if "object detection" in model_card.lower():
    model_type = "detection"
elif "classification" in model_card.lower():
    model_type = "classification"
elif "segmentation" in model_card.lower():
    model_type = "segmentation"
elif "generation" in model_card.lower() or "generate" in model_card.lower():
    model_type = "genai"

# Try to determine media type from model card
if "image" in model_card.lower():
    media_type = "image"
elif "text" in model_card.lower():
    media_type = "text"
elif "multimodal" in model_card.lower() or "vision-language" in model_card.lower():
    media_type = "multimodal"

# give an example of a model card -> adapter
ex1_repo = 'Salesforce/blip2-opt-2.7b'  # image captioning, prompt item with image
ex2_repo = 'facebook/detr-resnet-50'  # object detection, image item
ex3_repo = 'google/pegasus-xsum'  # text item, text item or prompt item with text

fxns_to_incl = ['load', 'prepare_item_func', 'predict']

# Get the template and render it with the required variables
template = ModelAdapterPrompts.get_template(MODEL_ADAPTER_TYPE)

# Prepare template variables based on the template type
template_variables = {}

if MODEL_ADAPTER_TYPE == "create_model_adapter":
    template_variables = {
        'fxns_to_incl': fxns_to_incl,
        'dtlpy_context': dtlpy_context,
        'ex1_repo': ex1_repo,
        'script_string1': script_string1,
        'ex2_repo': ex2_repo,
        'script_string2': script_string2,
        'ex3_repo': ex3_repo,
        'script_string3': script_string3,
        'model_card': model_card,
    }
elif MODEL_ADAPTER_TYPE == "create_hugging_adapter":
    # For hugging base adapter, we only need the example script and model card
    example_script = script_to_string(
        os.path.join(os.getcwd(), "generate/responses/detr-resnet-50_2025.03.19_14.44.08.py")
    )
    template_variables = {'example_script': example_script, 'model_card': model_card}
elif MODEL_ADAPTER_TYPE == "convert_to_hugging_base":
    template_variables = {
        'hugging_base_code': hugging_base_code,
        'example_adapter': example_adapter,
        'original_adapter': original_adapter,
    }

# Render the template with the appropriate variables
user_prompt = template.render(**template_variables)

response, usage = get_completion(user_prompt)
response = response.replace(r"```python", "")
response = response.replace(r"```", "")

responses_dir = os.path.join(os.getcwd(), "generate", "responses")
os.makedirs(responses_dir, exist_ok=True)
model_name = MODEL_REPO.split('/')[1]
responses_path = os.path.join(
    responses_dir, f"{model_name}_{media_type}_{model_type}_{time.strftime('%Y.%m.%d_%H.%M.%S')}.py"
)
with open(responses_path, "w") as f:
    f.write(response)

with open(usage_file, 'a') as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- {MODEL_REPO}: {usage}\n")

print(f"done! model adapter saved to {responses_path}")
