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


def create_model_entity(package: dl.Package) -> dl.Model:
    hugging = HuggingAdapter({})
    id2label = hugging.model.config.id2label
    model = package.models.create(model_name='facebook/detr-resnet-50-panoptic',
                                  description='facebook/detr-resnet-50-panoptic',
                                  tags=['pretrained', 'facebook', 'huggingface', 'panoptic'],
                                  dataset_id=None,
                                  scope='project',
                                  status='trained',
                                  labels=list(id2label.values()),
                                  configuration={'module_name': 'models.facebook_detr_resnet_50_panoptic',
                                                 'id_to_label_map': id2label,
                                                 'label_to_id_map': {v: k for k, v in id2label.items()}},
                                  project_id=package.project.id
                                  )
    return model


def script():
    import matplotlib.pyplot as plt
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)

    hugging = HuggingAdapter()

    # prepare image for the model
    inputs = hugging.feature_extractor(images=img,
                                       return_tensors="pt",
                                       pad_and_return_pixel_mask=False)

    # forward pass
    outputs = hugging.model(**inputs)

    # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = hugging.feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    # the segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    # retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb_to_id(panoptic_seg)
    plt.figure()
    plt.imshow(panoptic_seg_id)
