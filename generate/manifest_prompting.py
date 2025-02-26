import openai
import os
import time
from huggingface_hub import hf_hub_download

CARD_TYPE = 'url'
MODEL_REPO = 'nlpconnect/vit-gpt2-image-captioning'

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

usage_file = os.path.join(os.getcwd(), 'usage.txt')


# don't use gpt4, too $$$
def get_completion(prompt, model="gpt-4o-mini"):  # gpt-4o-mini-2024-07-18
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content, response.usage


# get the model card to be generated
if CARD_TYPE == 'file':
    model_card_path = hf_hub_download(repo_id=MODEL_REPO, filename='README.md')
    with open(model_card_path, 'r') as f:
        model_card = f.read()
else:
    model_card = rf"https://huggingface.co/{MODEL_REPO}"

# give an example of a model card -> adapter
ex1_hf_repo = 'Salesforce/blip2-opt-2.7b'
ex1_manifest = 'https://github.com/dataloop-ai-apps/huggingface-adapter/blob/APPS-1558-add-BLIP-2-model-adapter/adapters/blip_2/dataloop.json'

ex2_hf_repo = 'Salesforce/blip-image-captioning-base'
ex2_manifest = 'https://github.com/dataloop-ai-apps/huggingface-adapter/blob/APPS-1590-refactor-BLIP-model-adapter/adapters/blip_image_captioning_large/dataloop.json'

# fxn_to_include = ['load', 'prepare_item_func', 'predict', 'train', 'reformat_messages', 'compute_confidences']
fxns_to_incl = ['load', 'prepare_item_func', 'predict']

system_prompt = rf"""
Your task is to generate a json that creates a new application for the Dataloop platform. \\
The "name" should 
The application will be based on the {MODEL_REPO} model.\


Use the examples below to complete the task. \n\n

Model adapter example:\
 1: This is a json manifest for the {ex1_hf_repo} model created based on the model info here: {ex1_manifest}. \n \\
 2: This is a json manifest for the {ex2_hf_repo} model created based on the model info here: {ex2_manifest}. \n \\

The `HuggingAdapter` class should have the following methods: {fxns_to_incl}. \n 
Using the model info provided below, use the examples to generate the HuggingAdapter class script for the new \
model. Pay special attention to the code snippets in the model card, and include the relevant code in the "predict"\
 or "prepare_item_func" functions of the HuggingAdapter off of it. In the "load" function, 


Don't repeat this prompt and keep the response as concise as possible.\

Model Info: ```{model_card}```\
"""

response, usage = get_completion(system_prompt)
response = response.replace(r"```python", "")
response = response.replace(r"```", "")

with open(os.path.join(os.getcwd(), "responses", f"{MODEL_REPO.split('/')[1]}_{time.strftime('%Y.%m.%d_%H.%M.%S')}.py"),
          "w") as f:
    f.write(response)

with open(usage_file, 'a') as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- {MODEL_REPO}: {usage}\n")

print("done!")