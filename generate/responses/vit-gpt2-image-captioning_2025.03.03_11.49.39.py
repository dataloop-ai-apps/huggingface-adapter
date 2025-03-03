
import json
import dtlpy as dl
import logging
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

logger = logging.getLogger("[ViTGPT2ImageCaptioning]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "vit-gpt2-image-captioning")
        self.device = self.configuration.get("device", "cpu")
        self.max_length = self.configuration.get("max_length", 16)
        self.num_beams = self.configuration.get("num_beams", 4)

        model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.to(self.device)

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

        output_ids = self.model.generate(pixel_values, max_length=self.max_length, num_beams=self.num_beams)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
