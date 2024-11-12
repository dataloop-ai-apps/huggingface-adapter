import base64
from io import BytesIO
import json
from typing import List
import PIL
import dtlpy as dl
import logging
from transformers import AutoProcessor, Blip2ForConditionalGeneration

logger = logging.getLogger("[BLIP]")
CAPTIONING_PROMPT = "Caption this image."


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name", "blip-2")
        self.device = configuration.get("device", "cpu")
        self.conditioning = configuration.get("conditioning", False)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model.to(self.device)

    def prepare_item_func(item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.Item], **kwargs):
        for prompt_item in batch:
            prompt_txt, image_buffer = self.reformat_messages(
                prompt_item.to_messages(model_name=self.model_name)
            )
            prompt = "Question: {} Answer:".format(prompt_txt)
            encoding = self.processor(
                PIL.Image.open(image_buffer), prompt, return_tensors="pt"
            ).to(self.device)
            output = self.model.generate(**encoding)
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

    def reformat_messages(self, messages):
        def get_last_user_message(messages):
            for message in reversed(messages):
                if message.get("role") == "user":
                    return message
            raise ValueError("No message with role 'user' found")

        last_user_message = get_last_user_message(messages)
        for content in last_user_message["content"]:
            if content["type"] == "text":
                prompt_txt = content["text"]
            elif content["type"] == "image_url":
                base64_str = content["image_url"]["url"].split("base64,")[1]
                image_buffer = BytesIO(base64.b64decode(base64_str))

        return prompt_txt, image_buffer
