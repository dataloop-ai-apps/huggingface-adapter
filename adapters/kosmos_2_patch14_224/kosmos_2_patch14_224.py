import base64
import PIL
from io import BytesIO
from typing import List
import dtlpy as dl
import logging
import torch

from transformers import AutoProcessor, AutoModelForVision2Seq

logger = logging.getLogger("[Kosmos-2 Patch 14 224]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "kosmos-2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.model.to(self.device)
        self.max_new_tokens = self.configuration.get("max_new_tokens", 128)

    def prepare_item_func(self, item: dl.Item):
        if "image/" in item.mimetype:
            prompt_txt = item.description if item.description is not None else "An image of"
            image_buffer = item.download(save_locally=False)
        elif "application/json" in item.mimetype:
            prompt_item = dl.PromptItem.from_item(item)
            prompt_txt, image_buffer = self.reformat_messages(prompt_item.to_messages(model_name=self.model_name))
        else:
            raise ValueError("Item must be a either image or prompt item for grounded image captioning.")
        return item, prompt_txt, image_buffer

    def predict(self, batch: List[dl.Item], **kwargs):
        annotations = []
        for item, prompt_txt, image_buffer in batch:
            # Open image to get dimensions
            pil_image = PIL.Image.open(image_buffer)
            image_width, image_height = pil_image.size

            prompt_txt = f"<grounding> {prompt_txt}"
            encoding = self.processor(text=prompt_txt, images=PIL.Image.open(image_buffer), return_tensors="pt").to(
                self.device
            )
            generated_ids = self.model.generate(
                pixel_values=encoding["pixel_values"],
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=encoding["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text, entities = self.processor.post_process_generation(generated_text)

            if item.mimetype == "application/json":
                prompt_item = dl.PromptItem.from_item(item)
                prompt_item.add(
                    message={
                        "role": "assistant",
                        "content": [{"mimetype": dl.PromptType.TEXT, "value": processed_text}],
                    },
                    model_info={"name": self.model_name, "confidence": 1.0, "model_id": 1},
                )
            else:
                item.description = None
                item.update()
                item.description = f"{processed_text} (captioned by {self.model_name})"

                # Convert entities to Dataloop annotations
                entity_annotations = self.convert_entities_to_annotations(entities, image_width, image_height)
                annotations.append(entity_annotations)

        return annotations

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

        if prompt_txt is None:
            logging.warning("No text found in messages, using blank text.")
            prompt_txt = ""
        if not image_buffer:
            raise ValueError("No image found in messages.")

        return prompt_txt, image_buffer

    @staticmethod
    def convert_entities_to_annotations(entities, image_width, image_height):
        """
        Convert entities from Kosmos-2 format to Dataloop bounding box annotations.

        Args:
            entities: List of tuples in format [(entity_name, (start, end), [(x1_norm, y1_norm, x2_norm, y2_norm)])]
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            List of Dataloop bounding box annotations
        """
        annotations = dl.AnnotationCollection()

        for entity_name, (start, end), bboxes in entities:
            for bbox in bboxes:
                x1_norm, y1_norm, x2_norm, y2_norm = bbox

                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1_norm * image_width)
                y1 = int(y1_norm * image_height)
                x2 = int(x2_norm * image_width)
                y2 = int(y2_norm * image_height)

                # Create Dataloop bounding box annotation
                annotations.add(
                    dl.Box(
                        top=y1,
                        left=x1,
                        bottom=y2,
                        right=x2,
                        label=entity_name,
                        attributes={"text_start": start, "text_end": end, "confidence": 1.0},
                    )
                )
        return annotations
