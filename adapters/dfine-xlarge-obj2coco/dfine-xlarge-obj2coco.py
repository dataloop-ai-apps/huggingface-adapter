
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
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if 'image/' not in item.mimetype:
            raise ValueError("Item must be an image for object detection.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []

        for item in batch:
            try:
                # Download and process image
                image_buffer = item.download(save_locally=False)
                image = PIL.Image.open(image_buffer).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                # Perform inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process results
                target_sizes = torch.tensor([image.size[::-1]])
                results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

                # Create box annotations for each detection
                item_annotations = dl.AnnotationCollection()
                for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                    score, label = score.item(), label_id.item()
                    box = [round(i, 2) for i in box.tolist()]
                    left, top, right, bottom = box

                    # Create box annotation
                    item_annotations.add(
                        annotation_definition=dl.Box(
                            label=self.model.config.id2label[label], top=top, left=left, bottom=bottom, right=right
                        ),
                        model_info={
                            "name": self.model_name,
                            "model_id": self.model_name,
                            "confidence": round(score, 3),
                        },
                    )
                batch_annotations.append(item_annotations)

            except Exception as e:
                logger.error(f"Error processing item {item.id}: {str(e)}")

        return batch_annotations
