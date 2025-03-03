
import json
import torch
import PIL
import dtlpy as dl
import logging
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

logger = logging.getLogger("[Image Captioning]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "vit-gpt2-image-captioning")
        self.device = self.configuration.get("device", "cpu")
        self.max_length = self.configuration.get("max_length", 16)
        self.num_beams = self.configuration.get("num_beams", 4)

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        
        for prompt_key, prompt_content in prompts.items():
            image_buffer = None
            prompt_text = None
            
            for prompt_part in prompt_content:
                if "image" in prompt_part["mimetype"]:
                    image_url = prompt_part["value"]
                    item_id = image_url.split("/stream")[0].split("/items/")[-1]
                    image_buffer = dl.items.get(item_id=item_id).download(save_locally=False)
                elif "text" in prompt_part["mimetype"]:
                    prompt_text = prompt_part["value"]

            if image_buffer and prompt_text:
                ready_prompts.append((prompt_key, image_buffer, prompt_text))
            else:
                raise ValueError(f"{prompt_key} is missing either an image or a text prompt.")

        return ready_prompts

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            for prompt_key, image_buffer, prompt_txt in prompts:
                if isinstance(image_buffer, bytes):
                    image = PIL.Image.open(image_buffer).convert("RGB")
                else:
                    image = PIL.Image.open(image_buffer)

                pixel_values = self.feature_extractor(images=image, return_tensors='pt').pixel_values.to(self.device)
                output_ids = self.model.generate(pixel_values, max_length=self.max_length, num_beams=self.num_beams)
                response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                item_annotations.add(
                    annotation_definition=dl.FreeText(text=response),
                    prompt_id=prompt_key,
                    model_info={
                        'name': self.model_name,
                        'confidence': 1.0
                    }
                )
            annotations.append(item_annotations)
        return annotations
