import json

import PIL
import dtlpy as dl
from transformers import ViltProcessor, ViltForQuestionAnswering


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            image_buffer, prompt_text = None, None
            for prompt_part in prompt_content:
                if "image" in prompt_part["mimetype"]:
                    image_url = prompt_part["value"]
                    item_id = image_url.split("/stream")[0].split("/items/")[-1]
                    item = dl.items.get(item_id=item_id)
                    image_buffer = item.download(save_locally=False)
                elif "text" in prompt_part["mimetype"]:
                    prompt_text = prompt_part["value"]
            ready_prompts.append((prompt_key, image_buffer, prompt_text))

        return ready_prompts

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = []
            for prompt_key, image_buffer, prompt_text in prompts:
                encoding = self.processor(PIL.Image.open(image_buffer), prompt_text, return_tensors="pt")
                outputs = self.model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                response = self.model.config.id2label[idx]
                print("Response: {}".format(response))

                item_annotations.append({
                    "type": "text",
                    "label": "q",
                    "coordinates": response,
                    "metadata": {
                        "system": {"promptId": prompt_key},
                        "user": {
                            "annotation_type": "prediction",
                            "model": {
                                "name": self.model_name
                            }
                        }
                    }
                })
            annotations.append(item_annotations)
        return annotations


def model_creation(package: dl.Package):
    model = package.models.create(model_name='vilt_b32_finetuned_vqa',
                                  description='ask question about an image',
                                  tags=["hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'vilt_b32_finetuned_vqa.pt',
                                      "module_name": "models.vilt_b32_finetuned_vqa",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
