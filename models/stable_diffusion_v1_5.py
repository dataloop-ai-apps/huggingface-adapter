import datetime
import json
import os
import shutil
import dtlpy as dl
import torch
from diffusers import StableDiffusionPipeline

STREAM_URL = r"https://gate.dataloop.ai/api/v1/items/{}/stream"


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")

        self.model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                             torch_dtype=torch.float16)
        self.model.to("cuda")
        self.results_local_path = "stable_diffusion_v1_5_results"
        create_folder(self.results_local_path)

    def prepare_item_func(self, item: dl.Item):
        dataset_id = item.dataset_id
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            prompt_text = prompt_content[0]["value"]
            ready_prompts.append((prompt_key, prompt_text, dataset_id))
        return ready_prompts

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = []
            for prompt_key, prompt_text, dataset_id in prompts:
                image_result = self.model(prompt_text).images[0]

                image_result.show()
                image_result_path = os.path.join(self.results_local_path,
                                                 f"image_result_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg")

                image_result.save(image_result_path)
                dataset = dl.datasets.get(dataset_id=dataset_id)
                result_item_id = dataset.items.upload(local_path=image_result_path,
                                                      remote_path="stable_diffusion_v1_5_results").id
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
    model = package.models.create(model_name='stable-diffusion-v1-5',
                                  description='make image by text',
                                  tags=["hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'stable_diffusion_v1_5.pt',
                                      "module_name": "models.stable_diffusion_v1_5",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
