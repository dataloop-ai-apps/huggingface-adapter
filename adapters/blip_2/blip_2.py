import base64
from io import BytesIO
from typing import List
import PIL
import dtlpy as dl
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration

logger = logging.getLogger("[BLIP-2]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "blip-2")
        self.device = self.configuration.get("device", "cpu")
        self.conditioning = self.configuration.get("conditioning", False)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.Item], **kwargs):
        for prompt_item in batch:
            prompt_txt, image_buffer = self.reformat_messages(prompt_item.to_messages(model_name=self.model_name))
            if prompt_txt:
                encoding = self.processor(PIL.Image.open(image_buffer), prompt_txt, return_tensors="pt").to(self.device)
            else:
                encoding = self.processor(PIL.Image.open(image_buffer), return_tensors="pt").to(self.device)
            output = self.model.generate(**encoding)
            response = self.processor.decode(output[0], skip_special_tokens=True).strip()
            print("Response: {}".format(response))
            prompt_item.add(
                message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
                model_info={"name": self.model_name, "confidence": 1.0, "model_id": 1},
            )
        return []

    @staticmethod
    def reformat_messages(messages):
        def get_last_user_message(messages):
            for message in reversed(messages):
                if message.get("role") == "user":
                    return message
            raise ValueError("No message with role 'user' found")

        last_user_message = get_last_user_message(messages)

        prompt_txt = None
        image_buffer = None

        for content in last_user_message["content"]:

            content_type = content.get("type")

            if content_type == "text":
                # Concatenate multiple text contents with space
                new_text = content.get("text", "").strip()
                if new_text:
                    prompt_txt = f"{prompt_txt} {new_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url:
                    if image_buffer:
                        logger.error("Multiple images not supported, using only the first one")
                    else:
                        base64_str = content["image_url"]["url"].split("base64,")[1]
                        image_buffer = BytesIO(base64.b64decode(base64_str))

        if prompt_txt:
            prompt_txt = "Question: {} Answer:".format(prompt_txt)
        else:
            # If no text found, generates from the BOS token:
            # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BLIP-2/Chat_with_BLIP_2.ipynb
            logging.warning("No text found in messages, generating from the BOS (beginning-of-sequence) token.")
        if not image_buffer:
            raise ValueError("No image found in messages.")

        return prompt_txt, image_buffer
