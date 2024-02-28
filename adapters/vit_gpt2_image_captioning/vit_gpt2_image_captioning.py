import json
import os
import shutil
import dtlpy as dl
from PIL import Image
import logging
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

logger = logging.getLogger("[ViTGPT2ImageCaptioning]")


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        self.image_width = configuration.get("image_width", 512)
        self.image_height = configuration.get("image_height", 512)
        self.device = configuration.get("device")

        model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.to(self.device)
        self.gen_kwargs = {"max_length": configuration.get("max_length", 16),
                           "num_beams": configuration.get("num_beams", 4)}
        self.results_local_path = "vit_gpt2_image_captioning_results"
        create_folder(self.results_local_path)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

            prompt_image_found = False
            for prompt_part in questions:
                if "image" in prompt_part["mimetype"] and not prompt_image_found:
                    image_url = prompt_part["value"]
                    item_id = image_url.split("/stream")[0].split("/items/")[-1]
                    item = dl.items.get(item_id=item_id)
                    image_buffer = item.download(save_locally=False)
                    ready_prompts.append((prompt_key, image_buffer))
                    prompt_image_found = True
                else:
                    logger.warning(f"ViT GPT2 only accepts image prompts, ignoring the current prompt.")

            if not prompt_image_found:
                raise ValueError(f"{prompt_key} is missing image prompts.")

        return ready_prompts

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            prompts_zip = list(zip(*prompts))
            prompt_keys = prompts_zip[0]
            image_buffers = prompts_zip[1]
            preds = self.predict_step(image_buffers)

            for i, response in enumerate(preds):
                prompt_key = prompt_keys[i]
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

    def predict_step(self, image_buffers):
        images = []
        for image_buffer in image_buffers:
            i_image = Image.open(image_buffer)
            i_image = i_image.resize(size=(self.image_width, self.image_height))
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
