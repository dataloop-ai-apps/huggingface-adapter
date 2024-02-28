import dtlpy as dl
import torch
import numpy as np
import io

from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image


class HuggingAdapter:
    def __init__(self, configuration):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        self.model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    def predict(self, model_entity, batch, **kwargs):
        threshold = 0.7
        batch_annotations = list()
        for image in batch:
            image = Image.fromarray(image.astype('uint8')).convert("RGB")
            processed_outputs = self.make_prediction(image)

            # Visualize prediction
            collection = dl.AnnotationCollection()
            for label_id, label_name in model_entity.id_to_label_map.items():
                mask = processed_outputs == label_id
                if mask.any():
                    collection.add(annotation_definition=dl.Segmentation(geo=mask,
                                                                         label=label_name),
                                   model_info={'name': model_entity.name,
                                               'confidence': 1.,
                                               'model_id': model_entity.id,
                                               'dataset_id': model_entity.dataset_id})
                    batch_annotations.append(collection)
        return batch_annotations

    @staticmethod
    def rgb_to_id(color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def make_prediction(self, image):
        # prepare image for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # forward pass
        outputs = self.model(**inputs)

        # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = self.feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        # retrieve the ids corresponding to each mask
        panoptic_seg_id = self.rgb_to_id(panoptic_seg)
        return panoptic_seg_id

    def prepare_item_func(self, item: dl.Item):
        image = item.download(save_locally=False, to_array=True)
        return image
