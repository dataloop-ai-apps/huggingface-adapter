import openai
import os
import time
from openai_key import api_key
from huggingface_hub import hf_hub_download

CARD_TYPE = 'url'
MODEL_REPO = "dslim/bert-base-NER"
#'nlpconnect/vit-gpt2-image-captioning'
EXISTING_ADAPTER = "https://raw.githubusercontent.com/dataloop-ai-apps/huggingface-adapter/refs/heads/APPS-1558-add-BLIP-2-model-adapter/adapters/bert_base_ner/bert_base_ner.py"
# "https://raw.githubusercontent.com/dataloop-ai-apps/huggingface-adapter/refs/heads/APPS-1558-add-BLIP-2-model-adapter/adapters/vit_gpt2_image_captioning/vit_gpt2_image_captioning.py"

client = openai.OpenAI(api_key=api_key)
usage_file = os.path.join(os.getcwd(), 'usage.txt')


# don't use gpt4o, too $$$
def get_completion(prompt, model="gpt-4o-mini"):  # gpt-4o-mini-2024-07-18
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content, response.usage


# get the model card to be generated
if CARD_TYPE == 'file':
    model_card_path = hf_hub_download(repo_id=MODEL_REPO, filename='README.md')
    with open(model_card_path, 'r') as f:
        model_card = f.read()
else:
    model_card = rf"https://huggingface.co/{MODEL_REPO}"

# give an example of a model card -> adapter
ex1_repo = 'Salesforce/blip2-opt-2.7b'
ex1_adapter = 'https://raw.githubusercontent.com/dataloop-ai-apps/huggingface-adapter/refs/heads/APPS-1558-add-BLIP-2-model-adapter/adapters/blip_2/blip_2.py'

ex2_repo = 'Salesforce/blip-image-captioning-base'
ex2_adapter = 'https://raw.githubusercontent.com/dataloop-ai-apps/huggingface-adapter/refs/heads/APPS-1590-refactor-BLIP-model-adapter/adapters/blip_image_captioning_large/blip_image_captioning_large.py'

# fxn_to_include = ['load', 'prepare_item_func', 'predict', 'train', 'reformat_messages', 'compute_confidences']
fxns_to_incl = ['load', 'prepare_item_func', 'predict']

system_prompt = rf"""\
You are a Python machine learning developer working on a project that requires creating a model adapter for a\
 a machine learning model from HuggingFace.\\
 
 The model adapter code will be based on the code snippets available in HuggingFace model card, wrapped into a \
 Python class called `HuggingAdapter`. The script will import the `dtlpy` library as `dl`, and the \
 `HuggingAdapter` will inherit from the base class at `dl.BaseModelAdapter` from the `dtlpy` library, \
 linked here: https://github.com/dataloop-ai/dtlpy/blob/master/dtlpy/ml/base_model_adapter.py . \
  
 Your task is to refactor code that exists at the link for the "existing adapter" below. \
 No init function should be included, as the class is initialized with the `BaseModelAdapter` class. \
 Any fields from the `configuration` argument from the original `init` should be moved to the `load` function, \
 and retrieved from the class attribute `self.configuration`.\

Using the model info provided below, use the examples to refactor the HuggingAdapter class script for the new \
model. Pay special attention to the code snippets in the model card, and include the relevant code in the \
`prepare_item_func` and `predict` functions of `HuggingAdapter`. \
 
 If the model is a large language model or large multimodal model, the `prepare_item_func` function should \\
 convert the `dl.Item` object to a `dl.PromptItem` object, and add prompts and messages according to the code\
 in the example adapters listed below. \
 For the `predict` function, the model predictions should be created as annotations using the `dtlpy` library \
 to build and with code that closely resembles the predict functions from the "existing model adapter code".

Don't repeat this prompt and keep the response as concise as possible.\

Existing model adapter code: {EXISTING_ADAPTER} \n\n
Original model card for the model adapter: {model_card} \n\n
The `HuggingAdapter` class should have the methods: {fxns_to_incl} \n\n 

Few shot examples:\
 1: This is an adapter for the {ex1_repo} model created based on the model info here: {ex1_adapter}. \n \\
 2: This is an adapter for the {ex2_repo} model created based on the model info here: {ex2_adapter}. \n \\
"""

response, usage = get_completion(system_prompt)
print(response)

response = response.replace(r"```python", "")
response = response.replace(r"```", "")

model_repo_name = MODEL_REPO.split('/')[1]
os.makedirs(os.path.join(os.getcwd(), "responses", model_repo_name), exist_ok=True)
with open(
    os.path.join(
        os.getcwd(), "responses", model_repo_name, f"{model_repo_name}_{time.strftime('%Y.%m.%d_%H.%M.%S')}.py"
    ),
    "w",
) as f:
    f.write(response)

print("logging gpt usage...")
with open(usage_file, 'a') as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- {model_repo_name}: {usage}\n")

print("done!")
