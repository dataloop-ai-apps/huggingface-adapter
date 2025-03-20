
import base64
from io import BytesIO
from typing import List
import torch
import PIL
import dtlpy as dl
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger("[DETR]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "detr")
        self.device = self.configuration.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.Item], **kwargs):
        for prompt_item in batch:
            prompt_txt, image_buffer = self.reformat_messages(
                prompt_item.to_messages(model_name=self.model_name)
            )
            inputs = self.processor(images=PIL.Image.open(image_buffer).convert('RGB'), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            target_sizes = torch.tensor([PIL.Image.open(image_buffer).size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detections.append({
                    "label": self.model.config.id2label[label.item()],
                    "confidence": round(score.item(), 3),
                    "box": box
                })
                logger.debug(f"Detected {self.model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.JSON, "value": detections}],
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

        prompt_txt = None
        image_buffer = None

        for content in last_user_message["content"]:
            content_type = content.get("type")

            if content_type == "text":
                new_text = content.get("text", "").strip()
                if new_text:
                    prompt_txt = f"{prompt_txt} {new_text}".strip() if prompt_txt else new_text

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url:
                    base64_str = content["image_url"]["url"].split("base64,")[1]
                    image_buffer = BytesIO(base64.b64decode(base64_str))

        if prompt_txt is None:
            prompt_txt = "What objects are in this image?"

        if image_buffer is None:
            raise ValueError("No image found in messages.")

        return prompt_txt, image_buffer

    @staticmethod
    def get_last_prompt_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")
