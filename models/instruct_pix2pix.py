import datetime
import json
import os
import shutil
import PIL
import dtlpy as dl
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

STREAM_URL = r"https://gate.dataloop.ai/api/v1/items/{}/stream"


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")

        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                            torch_dtype=torch.float16,
                                                                            safety_checker=None)
        self.model.to("cuda")
        self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
        self.results_local_path = "instruct_pix2pix_results"
        create_folder(self.results_local_path)

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
                    dataset_id = item.dataset_id
                    image_buffer = item.download(save_locally=False)
                elif "text" in prompt_part["mimetype"]:
                    prompt_text = prompt_part["value"]
            ready_prompts.append((prompt_key, image_buffer, prompt_text, dataset_id))

        return ready_prompts

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = []
            for prompt_key, image_buffer, prompt_text, dataset_id in prompts:
                image_result = self.model(prompt_text, image=PIL.Image.open(image_buffer), num_inference_steps=5,
                                          image_guidance_scale=1).images[0]
                image_result.show()
                image_result_path = os.path.join(self.results_local_path,
                                                 f"image_result_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg")

                image_result.save(image_result_path)
                dataset = dl.datasets.get(dataset_id=dataset_id)
                result_item_id = dataset.items.upload(local_path=image_result_path,
                                                      remote_path="instruct_pix2pix_results").id
                os.remove(image_result_path)

                stream_url = STREAM_URL.format(str(result_item_id))
                item_annotations.append({
                    "type": "binary",
                    "label": "q",
                    "coordinates": stream_url,
                    "metadata": {
                        "system": {"promptId": prompt_key},
                        "user": {
                            "stream": True,
                            "model": {
                                "name": self.model_name
                            }
                        }
                    }
                })
            annotations.append(item_annotations)
        return annotations


def model_creation(package: dl.Package):
    model = package.models.create(model_name='instruct-pix2pix',
                                  description='change image by text',
                                  tags=["hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'pix2pix.pt',
                                      "module_name": "models.instruct_pix2pix",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
