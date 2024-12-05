import json
import torch
import PIL
import dtlpy as dl
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger("[BLIP]")
CAPTIONING_PROMPT = "Caption this image."


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "blip-2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conditioning = self.configuration.get("conditioning", False)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            for prompt_key, image_buffer, prompt_txt in prompts:
                encoding = self.processor(PIL.Image.open(image_buffer), prompt_txt, return_tensors='pt').to(self.device)
                output = self.model.generate(**encoding)
                response = self.processor.decode(output[0], skip_special_tokens=True)
                print("Response: {}".format(response))
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
