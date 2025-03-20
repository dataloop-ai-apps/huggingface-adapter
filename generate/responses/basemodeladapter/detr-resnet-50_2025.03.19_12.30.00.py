# this one was edited by hand to work

import PIL
import torch
import logging
import dtlpy as dl

from typing import List
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger("[HF-DETR-RESNET-50]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "detr-resnet-50")
        self.device = self.configuration.get("device", "cpu")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if 'image/' not in item.mimetype:
            raise ValueError("Item must be an image for object detection.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []

        for item in batch:
            # Download and process image
            image_buffer = item.download(local_path=None, overwrite=True)
            inputs = self.processor(images=PIL.Image.open(image_buffer).convert('RGB'), return_tensors="pt").to(
                self.device
            )
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([PIL.Image.open(image_buffer).size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Create box annotations for each detection
            item_annotations = dl.AnnotationCollection()
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                # DETR returns boxes in [x0, y0, x1, y1] format
                # Convert to [left, top, right, bottom] format
                left, top, right, bottom = box

                # Create box annotation
                item_annotations.add(
                    annotation_definition=dl.Box(
                        label=self.model.config.id2label[label.item()], top=top, left=left, bottom=bottom, right=right
                    ),
                    model_info={
                        "name": "detr-resnet-50",
                        "model_id": "detr-resnet-50",
                        "confidence": round(score.item(), 3),
                    },
                )
            batch_annotations.append(item_annotations)

        return batch_annotations
