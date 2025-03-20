# this script was edited by hand to work appropriately

import PIL
import torch
import logging
import dtlpy as dl

from typing import List
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger("[HF-DETR-RESNET-50]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "detr")
        self.device = self.configuration.get("device", "cpu")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if 'image/' not in item.mimetype:
            raise ValueError("Item must be an image for object detection.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        annotations = []
        for item in batch:
            image_buffer = item.download(local_path=None, overwrite=True)
            inputs = self.processor(images=PIL.Image.open(image_buffer).convert('RGB'), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([PIL.Image.open(image_buffer).size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                annotations.append(
                    {
                        "label": self.model.config.id2label[label.item()],
                        "confidence": round(score.item(), 3),
                        "box": box,
                    }
                )
        return annotations