import json
import torch
import PIL
import dtlpy as dl
import logging
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("[UForm]")
CAPTIONING_PROMPT = "Caption this image."


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        self.device = configuration.get("device")
        self.captioning = configuration.get("captioning", False)

        self.model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

            image_buffer = None
            prompt_text = CAPTIONING_PROMPT if self.captioning else None
            prompt_image_found = False
            prompt_text_found = self.captioning
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
                    logger.warning(f"UForm only accepts text and image prompts, "
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
                encoding = self.processor(text=[prompt_txt], images=[PIL.Image.open(image_buffer)], return_tensors='pt')
                encoding = encoding.to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        **encoding,
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=256,
                        eos_token_id=151645,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                prompt_len = encoding["input_ids"].shape[1]
                response = self.processor.batch_decode(output[:, prompt_len:-1])[0]
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
