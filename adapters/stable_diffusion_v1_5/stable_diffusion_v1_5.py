import datetime
import json
import os
import shutil
import dtlpy as dl
import logging
from diffusers import StableDiffusionPipeline
import torch

# STREAM_URL = r"https://gate.dataloop.ai/api/v1/items/{}/stream"
logger = logging.getLogger("[Stable Diffusion v1.5]")


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.results_local_path = "stable_diffusion_v1_5_results"
        create_folder(self.results_local_path)

    def prepare_item_func(self, item: dl.Item):
        dataset_id = item.dataset_id
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

            prompt_text_found = False
            for prompt_part in questions:
                if "text" in prompt_part["mimetype"]:
                    prompt_text = prompt_part["value"]
                    ready_prompts.append((prompt_key, prompt_text, dataset_id))
                    prompt_text_found = True
                    break
                else:
                    logger.warning(f"Stable Diffusion v1.5 only accepts text prompts, ignoring the current prompt.")

            if not prompt_text_found:
                raise ValueError(f"{prompt_key} is missing text prompts.")

        return ready_prompts

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_text, dataset_id in prompts:
                image_result = self.model(prompt_text).images[0]

                image_result.show()
                image_result_path = os.path.join(self.results_local_path,
                                                 f"image_result_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg")

                image_result.save(image_result_path)
                dataset = dl.datasets.get(dataset_id=dataset_id)
                result_item: dl.Item = dataset.items.upload(
                    local_path=image_result_path,
                    remote_path="stable_diffusion_v1_5_results"
                )
                os.remove(image_result_path)

                stream_url = result_item.stream
                item_annotations.add(
                    annotation_definition=dl.RefImage(ref=stream_url, mimetype=result_item.mimetype),
                    prompt_id=prompt_key,
                    model_info={
                        "name": logger.name.strip('[]'),
                        "confidence": 1.0
                    }
                )
            annotations.append(item_annotations)

        return annotations
