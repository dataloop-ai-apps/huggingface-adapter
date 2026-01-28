import json
import PIL
import dtlpy as dl
import torch
import logging
from transformers import ViltProcessor, ViltForQuestionAnswering

logger = logging.getLogger("[ViLT B32 Finetuned VQA]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

            image_buffer, prompt_text = None, None
            prompt_image_found, prompt_text_found = False, False
            for prompt_part in questions:
                if "image" in prompt_part["mimetype"] and not prompt_image_found:
                    image_url = prompt_part["value"]
                    item_id = image_url.split("/stream")[0].split("/items/")[-1]
                    item = dl.items.get(item_id=item_id)
                    image_buffer = item.download(save_locally=False)
                    prompt_image_found = True
                elif "text" in prompt_part["mimetype"] and not prompt_text_found:
                    prompt_text = prompt_part["value"]
                    prompt_text_found = True
                else:
                    logger.warning(f"ViLT B32 Finetuned Vqa only accepts text and image prompts, "
                                   f"ignoring the current prompt.")

                # Break loop after all inputs received
                if prompt_image_found and prompt_text_found:
                    break

            if prompt_image_found and prompt_text_found:
                ready_prompts.append((prompt_key, image_buffer, prompt_text))
            else:
                raise ValueError(f"{prompt_key} is missing either an image or a text prompt.")

        return ready_prompts

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            for prompt_key, image_buffer, prompt_txt in prompts:
                encoding = self.processor(PIL.Image.open(image_buffer), prompt_txt, return_tensors="pt").to(self.device)
                outputs = self.model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                response = self.model.config.id2label[idx]
                print("Response: {}".format(response))
                item_annotations.add(
                    annotation_definition=dl.FreeText(text=response),
                    prompt_id=prompt_key,
                    model_info={
                        "name": logger.name.strip('[]'),
                        "confidence": 1.0
                    }
                )
            annotations.append(item_annotations)
        return annotations
