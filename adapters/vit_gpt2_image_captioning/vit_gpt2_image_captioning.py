import datetime
import json
import os
import shutil
import dtlpy as dl
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = "cuda"
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
            if "image" in prompt_content[0]["mimetype"]:
                image_url = prompt_content[0]["value"]
                item_id = image_url.split("/stream")[0].split("/items/")[-1]
                item = dl.items.get(item_id=item_id)
                image_buffer = item.download(save_locally=False)
                ready_prompts.append((prompt_key, image_buffer))

        return ready_prompts

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
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


def model_creation(package: dl.Package):
    model = package.models.create(model_name='vit-gpt2-image-captioning',
                                  description='Textual description for the image',
                                  tags=["hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'vit-gpt2-image-captioning.pt',
                                      "module_name": "models.vit_gpt2_image_captioning",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
