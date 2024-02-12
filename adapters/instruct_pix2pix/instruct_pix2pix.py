import datetime
import json
import os
import shutil
import PIL
import dtlpy as dl
import logging
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# STREAM_URL = r"https://gate.dataloop.ai/api/v1/items/{}/stream"
logger = logging.getLogger("[InstructPix2Pix]")


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        self.image_width = configuration.get("image_width", 512)
        self.image_height = configuration.get("image_height", 512)

        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            # torch_dtype=torch.float32,
            safety_checker=None
        )
        self.model.to("cuda")
        self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
        self.results_local_path = "instruct_pix2pix_results"
        create_folder(self.results_local_path)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        prompts = buffer["prompts"]
        ready_prompts = []
        for prompt_key, prompt_content in prompts.items():
            questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

            image_buffer, prompt_text, dataset_id = None, None, None
            prompt_image_found, prompt_text_found = False, False
            for prompt_part in questions:
                if "image" in prompt_part["mimetype"] and not prompt_image_found:
                    image_url = prompt_part["value"]
                    item_id = image_url.split("/stream")[0].split("/items/")[-1]
                    item = dl.items.get(item_id=item_id)
                    dataset_id = item.dataset_id
                    image_buffer = item.download(save_locally=False)
                    prompt_image_found = True
                elif "text" in prompt_part["mimetype"] and not prompt_text_found:
                    prompt_text = prompt_part["value"]
                    prompt_text_found = True
                else:
                    logger.warning(f"Pix2Pix only accepts text and image prompts, ignoring the current prompt.")

                # Break loop after all inputs received
                if prompt_image_found and prompt_text_found:
                    break

            if prompt_image_found and prompt_text_found:
                ready_prompts.append((prompt_key, image_buffer, prompt_text, dataset_id))
            else:
                raise ValueError(f"{prompt_key} is missing either an image or a text.")

        return ready_prompts

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for prompts in batch:
            item_annotations = dl.AnnotationCollection()
            for prompt_key, image_buffer, prompt_text, dataset_id in prompts:
                image = PIL.Image.open(image_buffer)
                image = image.resize(size=(self.image_width, self.image_height))
                image_result = self.model(prompt_text, image=image, num_inference_steps=5,
                                          image_guidance_scale=1).images[0]
                image_result.show()
                image_result_path = os.path.join(self.results_local_path,
                                                 f"image_result_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg")

                image_result.save(image_result_path)
                dataset = dl.datasets.get(dataset_id=dataset_id)
                result_item = dataset.items.upload(local_path=image_result_path,
                                                   remote_path="instruct_pix2pix_results")
                os.remove(image_result_path)

                stream_url = result_item.stream
                item_annotations.add(annotation_definition=dl.RefImage(ref=stream_url, mimetype="image/png"),
                                     prompt_id=prompt_key,
                                     model_info={
                                         'name': self.model_name,
                                         'confidence': 1.0
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
