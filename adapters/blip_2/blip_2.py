import base64
from io import BytesIO
from typing import List
import PIL
import dtlpy as dl
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

logger = logging.getLogger("[BLIP-2]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "blip-2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conditioning = self.configuration.get("conditioning", False)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.PromptItem], **kwargs):
        for prompt_item in batch:
            prompt_txt, image_buffer = HuggingAdapter.reformat_messages(
                prompt_item.to_messages(model_name=self.model_name)
            )
            encoding = self.processor(
                PIL.Image.open(image_buffer).convert('RGB'), prompt_txt, return_tensors="pt"
            ).to(self.device)
            
            output = self.model.generate(**encoding, max_new_tokens=50)
            response = self.processor.decode(
                output[0], skip_special_tokens=True
            ).strip()
            print("Response: {}".format(response))
            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": response}],
                },
                model_info={
                    "name": self.model_name,
                    "confidence": 1.0,
                    "model_id": self.model_entity.id,
                },
            )
        return []
    
    def get_last_prompt_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")
        
    def reformat_messages(messages):
        # In case of multiple messages, 
        # we assume the last user message contains the image of interest

        last_user_message = HuggingAdapter.get_last_prompt_message(messages)
        
        prompt_txt = None
        image_buffer = None
        
        # The last user message may contain multiple contents, 
        # such as a text component and an image component
        # or multiple text components (e.g., multiple questions)
        for content in last_user_message["content"]:
            
            content_type = content.get("type", None)
            if content_type is None:
                raise ValueError("Message content type not found")
            
            if content_type == "text":
                # Concatenate multiple text contents with space
                new_text = content.get("text", "").strip()
                if new_text:
                    if prompt_txt is None:
                        prompt_txt = new_text
                    else:
                        prompt_txt = f"{prompt_txt} {new_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url is not None:
                    if image_buffer is not None: # i.e., we previously found an image
                        raise ValueError("Multiple images not supported")
                    else:
                        base64_str = content["image_url"]["url"].split("base64,")[1]
                        image_buffer = BytesIO(base64.b64decode(base64_str))
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        
        if prompt_txt is None:
            prompt_txt = "What is in this image?"
        prompt_txt = "Question: {} Answer:".format(prompt_txt)

        if not image_buffer:
            raise ValueError("No image found in messages.")

        return prompt_txt, image_buffer
