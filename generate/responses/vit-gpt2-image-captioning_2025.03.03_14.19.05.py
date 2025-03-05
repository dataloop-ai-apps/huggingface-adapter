
import base64
from io import BytesIO
from typing import List
import torch
import PIL
import dtlpy as dl
import logging
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

logger = logging.getLogger("[ViTGPT2ImageCaptioning]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "vit-gpt2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.to(self.device)
        self.gen_kwargs = {
            "max_length": self.configuration.get("max_length", 16),
            "num_beams": self.configuration.get("num_beams", 4)
        }

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.Item], **kwargs):
        for prompt_item in batch:
            prompt_txt, image_buffer = self.reformat_messages(
                prompt_item.to_messages(model_name=self.model_name)
            )
            if image_buffer is None:
                raise ValueError("No image found in messages.")
            
            i_image = PIL.Image.open(image_buffer).convert("RGB")
            pixel_values = self.feature_extractor(images=[i_image], return_tensors="pt").pixel_values.to(self.device)
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            logger.debug("Response: {}".format(response))
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

    @staticmethod
    def reformat_messages(messages):
        last_user_message = HuggingAdapter.get_last_prompt_message(messages)
        image_buffer = None

        for content in last_user_message["content"]:
            content_type = content.get("type")
            if content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url:
                    base64_str = image_url.split("base64,")[1]
                    image_buffer = BytesIO(base64.b64decode(base64_str))
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        return None, image_buffer

    @staticmethod
    def get_last_prompt_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")
