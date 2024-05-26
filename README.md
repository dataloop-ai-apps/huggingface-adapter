# huggingface-adapter

Use this repo to create Dataloop Model Adapters for Hugging Face Conversational models.

## Usage

### Model Selection

From [Hugging Face](https://huggingface.co/models?pipeline_tag=conversational&sort=downloads) select the model you desire. For this example, we will use [DialoGPT-Large](https://huggingface.co/microsoft/DialoGPT-large?text=Hi.).

### Create a HuggingAdapter

In the ```models``` directory, create a new python file that will contain a ```HuggingAdapter```. In this example, we create ```models/dialogpt-large.py```  which is initialized as:

```python
import dtlpy as dl

class HuggingAdapter:
```

### Find the Model and Tokenizer

From the Hugging Face page of the model you are working with, find out which type of transformers model and tokenizer they use. In the case of DialoGPT, we need ```AutoModelForCausalLLM``` and ```AutoTokenizer```.
In the initialization for the ```HuggingAdapter``` instantiate ```self.tokenizer``` and ```self.model``` with the required methods and inputs to correctly load pretrained weights:

```python
import dtlpy as dl
from transformers import AutoModelForCausalLLM, AutoTokenizer

class HuggingAdapter:
    def __init__(self, configuration):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
```

If you wish to use any other configuration, it can be read from the ```configuration``` input, which will be a dictionary created when the model is uploaded (see below)

### Implement text generation

Implement the ```predict``` function for ```HuggingAdapter```, which receives a batch of inputs ```batch``` and customizable ```**kwargs```.
The ```batch``` will be a list of inputs. Each input is a dictionary of the format:

```json
{
  "shebang": "dataloop",
  "metadata": {
    "dltype": "prompt"
  },
  "prompts": {
    "prompt1": {
      "question-1": {
        "mimetype": "application/text",
        "value": "<first-question>"
      },
      "question-2": {
        "mimetype": "application/text",
        "value": "<second-question>"
      },
      ...
    },
    
    "prompt2": {
      "question-1": {
        "mimetype": "application/text",
        "value": "<first-question>"
      },
      "question-2": {
        "mimetype": "application/text",
        "value": "<second-question>"
      },
      ...
    },
    ...
  }
}
```

Each ```prompt``` contains a sequence of questions that will be sent to the model. Each input can have multiple prompts. Each question should be answered with a dictionary with the following format:

```json
{
"type": "text",
"label": "q",
"coordinates": <response>,
"metadata": {
    "system": {"promptId": <prompt-key>},
    "user": {
        "annotation_type": "prediction",
        "model": {
            "name": "<model-name>",
            "confidence": <confindence_computation()>
            }
        }
   }
}
```

where ```<response>``` is the model generated response for a a specific answer, ```<prompt-key>``` is the prompt key under which the question originally was, ```<model-name>``` is the name of the model that generated the response and ```<confidence_computation>``` is a user implemented method to compute the model's confidence in the answer.

When implementing the ```predict``` function, keep in mind that you will have to loop over a list of inputs in the above format and for each generated answer, append them to a list of answers for a specific input and then append that list of answers to the list of answers for the whole batch. The answer generation should conform with the one presented in your model's Hugging Face page. In the case of ```DialoGPT-Large``` we have:

```python
    def predict(self, batch, **kwargs):
        annotations = []  # This list receives the annotations for the whole batch
        for item in batch:  # Looping over all the items in the batch
            prompts = item["prompts"]  # For each item we first extract the dictionary containing the prompts
            item_annotations = []  # The list that receives the answers specific to this item
            for prompt_key, prompt_content in prompts.items():  # Looping over each prompt in an item
                chat_history_ids = torch.tensor([])
                for question in prompt_content.values():
                    print(f"User: {question['value']}")
                    new_user_input_ids = self.tokenizer.encode(question["value"] + self.tokenizer.eos_token,
                                                               return_tensors='pt')  # Tokenizing the question
                    # Generation according to the model's Hugging Face page:
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) \
                        if len(chat_history_ids) else new_user_input_ids
                    chat_history_ids = self.model.generate(bot_input_ids, max_new_tokens=1000, do_sample=True,
                                                           pad_token_id=self.tokenizer.eos_token_id, top_k=self.top_k)
                    response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                                     skip_special_tokens=True)
                    print("Response: {}".format(response))
                    # Now the answer for the specific question is appended to the item annotations:
                    item_annotations.append({
                        "type": "text",
                        "label": "q",
                        "coordinates": response,
                        "metadata": {
                            "system": {"promptId": prompt_key},
                            "user": {
                                "annotation_type": "prediction",
                                "model": {
                                    "name": "DialoGPT-Large",
                                    "confidence": self.compute_confidence(new_user_input_ids)
                                    }
                                }}
                        })
            # Append the item annotations with the answers to one item to the annotations
            annotations.append(item_annotations)
        return annotations             
```

### Create the huggingface-adapter Package

If you already have this package created, you can skip to the next step to proceed with creating the specific model. If this is the first Hugging Face model adapter you create, open the ```creation.py``` script and adapt line ```41``` to use the project id of your Dataloop project.
After that, run the ```creation.py``` script.

### Create the model

Write a script to run a model creation function like this one:

```python
package = dl.packages.get(package_id=<id-of-the-package-you-created-in-the-last-step>)
model = package.models.create(model_name=<model-name>,
                              description=<model-description>
                              tags=['llm', 'pretrained', "hugging-face"],
                              dataset_id=None,
                              status='pre-trained',
                              scope='project',
                              configuration={
                                  'weights_filename': <filename-for-the-weights-file>,
                                  "module_name": "models.<name-of-the-file-you-created>",
                                  ...},
                              project_id=package.project.id
                              )
```

Notice some details: in the ```configuration``` you must add the ```module_name``` key with a value in the format ```models.<name>``` where ```<name>``` is the name of the file containing the HuggingAdapter you created. In the case of DialoGPT-Large, we have created ```models/dialogpt-large.py```, thus the configuration must be ```"module_name": "models.dialogpt-large"```.
Run the model creation script and your model will appear in the model management tab of your project:

![image](https://github.com/dataloop-ai-apps/huggingface-adapter/assets/124260926/ad10505a-7aa1-4f1f-8685-58011d1e7fa4)

