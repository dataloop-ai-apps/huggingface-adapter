
import PIL
import torch
import logging
import dtlpy as dl
from typing import List
from transformers import DFineForObjectDetection, AutoImageProcessor

logger = logging.getLogger("[D-FINE]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "d-fine")
        self.device = self.configuration.get("device", "cpu")
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if 'image/' not in item.mimetype:
            raise ValueError("Item must be an image for object detection.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []

        for item in batch:
            image_buffer = item.download(local_path=None, overwrite=True)
            open_image = PIL.Image.open(image_buffer)
            inputs = self.processor(images=open_image.convert('RGB'), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([open_image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

            item_annotations = dl.AnnotationCollection()
            for result in results:
                for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    left, top, right, bottom = box

                item_annotations.add(
                    annotation_definition=dl.Box(
                        label=self.model.config.id2label[label_id.item()], top=top, left=left, bottom=bottom, right=right
                    ),
                    model_info={
                        "name": self.model_name,
                        "model_id": self.model_name,
                        "confidence": round(score.item(), 3),
                    },
                )
            batch_annotations.append(item_annotations)

        return batch_annotations
