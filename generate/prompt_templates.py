"""
Module containing prompt templates and utilities for generating model adapters.

This module provides a collection of prompt templates used to guide LLM responses
when generating model adapters from Hugging Face models.
"""

import jinja2
from typing import Dict

# Template for generating model adapters
MODEL_ADAPTER_PROMPT_STR = """
You are an AI assistant specialized in Python development for Machine Learning and GenAI. 

Your task is to generate code that creates a new Python class called \
`HuggingAdapter`. This class should inherit from the `BaseModelAdapter` class in the dtlpy library. \
(link: https://github.com/dataloop-ai/dtlpy/blob/master/dtlpy/ml/base_model_adapter.py). \
Use the example below to complete the task. 

The `HuggingAdapter` class should have at least the following methods: {{ fxns_to_incl }}. 
Using the model info provided below, use the examples to generate the HuggingAdapter class script for the new \
model. Pay special attention to the code snippets in the model card, and include the relevant code in the `predict`\
 or `prepare_item_func` functions of the HuggingAdapter off of it.  \
No init function should be included, as the class is initialized with the `BaseModelAdapter` class.  \
The load function should include loading configurations from `self.configuration.get`.  \
`prepare_item_func` should be simple and checks whether the item mimetype has the components required to 
run the model correctly. For example, for a computer vision model, the item mimetype should be `image`,  \
using the regular `dl.Item` entity.  \
For a NLP model, the item mimetype should be `text` and include text only.  \
For a multimodal model, include in `prepare_item_func` the `dl.PromptItem` entity, 
with the relevant mimetype components required to run the model correctly.  \

Make sure you include all the relevant imports for the code to run.  \
Don't include any imports that are not used in the code.  \
Don't repeat this prompt and keep the response as concise as possible.  \

Create dltpy entities according to the dtlpy python sdk. Pay special attention to dl.PromptItem.
Here is the relevant code context: {{ dtlpy_context }}  


Model adapter example:\
 1: This is an adapter for the {{ ex1_repo }} model: \n \\\
 {{ script_string1 }}
 
 2: This is an adapter for the {{ ex2_repo }} model: \n \\\
 {{ script_string2 }}\
 
 \n\n\
 Info for the model to create an adapter: ```{{ model_card }}```\\
\n\n\
"""

# Template for generating model adapter scripts based on hugging_base.py
HUGGING_BASE_PROMPT_STR = """
You are an AI assistant specialized in Python development for Machine Learning and GenAI.

Your task is to generate a Python script that creates a new class called `HuggingAdapter` 
inheriting from the `HuggingBase` class. The script should be similar to the example below.

The script should:
1. Import the necessary dependencies including `HuggingBase` and the specific model's processor and model classes
2. Create a `HuggingAdapter` class that inherits from `HuggingBase`
3. Implement the `load_model_and_processor` method to load the specific model and processor
4. Include a `get_cmd` method that returns the command to run the script

Here is an example of how the script should look:
{{ example_script }}

Now, create a similar script for the following model:
{{ model_card }}

Make sure to:
1. Use the correct model name and processor/model classes from the model card
2. Keep the code concise and focused on the essential implementation
3. Include all necessary imports
4. Follow the same structure as the example

Don't repeat this prompt and keep the response as concise as possible.
"""

# Dictionary mapping template names to their content
PROMPT_TEMPLATES = {
    "model_adapter_prompt": MODEL_ADAPTER_PROMPT_STR,
    "hugging_base_adapter_prompt": HUGGING_BASE_PROMPT_STR,
}

# Create Jinja templates once
TEMPLATES: Dict[str, jinja2.Template] = {name: jinja2.Template(template) for name, template in PROMPT_TEMPLATES.items()}


class ModelAdapterPrompts:
    """
    A class providing access to model adapter prompt templates.

    This class serves as an interface to access and render various prompt templates
    used in the model adapter generation process. Templates are accessed either through
    attribute access or the get_template class method.

    Attributes:
        None

    Methods:
        __getattr__(name: str) -> str: Dynamically retrieves prompt template strings by name
        get_template(name: str) -> jinja2.Template: Retrieves compiled Jinja templates by name
    """

    def __getattr__(self, name: str) -> str:
        """
        Get the Jinja template by name

        Args:
            name (str): Name of the prompt template to retrieve

        Returns:
            str: The prompt template string

        Raises:
            AttributeError: If the requested template name doesn't exist
        """
        if name in PROMPT_TEMPLATES:
            return PROMPT_TEMPLATES[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    @classmethod
    def get_template(cls, name: str) -> jinja2.Template:
        """
        Get the compiled Jinja template by name.

        Args:
            name (str): Name of the template to retrieve

        Returns:
            jinja2.Template: The compiled Jinja template object

        Raises:
            KeyError: If the requested template name doesn't exist
        """
        return TEMPLATES[name]
