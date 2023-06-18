import dtlpy as dl
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, DetrForObjectDetection


class HuggingAdapter:
    def __init__(self, configuration):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/detr-resnet-101')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')

    def predict(self, model_entity, batch, **kwargs):
        threshold = 0.7
        batch_annotations = list()
        for image in batch:
            image = Image.fromarray(image.astype('uint8')).convert("RGB")
            processed_outputs = self.make_prediction(image)

            # Visualize prediction
            keep = processed_outputs["scores"] > threshold
            boxes = processed_outputs["boxes"][keep].tolist()
            scores = processed_outputs["scores"][keep].tolist()
            labels = processed_outputs["labels"][keep].tolist()
            labels = [self.model.config.id2label[x] for x in labels]

            for score, (xmin, ymin, xmax, ymax), label in zip(scores, boxes, labels):
                collection = dl.AnnotationCollection()
                collection.add(annotation_definition=dl.Box(top=ymin,
                                                            bottom=ymax,
                                                            left=xmin,
                                                            right=xmax,
                                                            label=label),
                               model_info={'name': model_entity.name,
                                           'confidence': score,
                                           'model_id': model_entity.id,
                                           'dataset_id': model_entity.dataset_id})
                batch_annotations.append(collection)
        return batch_annotations

    def make_prediction(self, img):
        inputs = self.feature_extractor(img, return_tensors="pt")
        outputs = self.model(**inputs)
        img_size = torch.tensor([tuple(reversed(img.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)
        return processed_outputs[0]


def create_model_entity():
    package = dl.packages.get(package_name='hugging-face')
    project = dl.projects.get(project_name='Hugging Face')
    hugging = HuggingAdapter()
    id2label = hugging.model.config.id2label
    model = package.models.create(model_name='facebook/detr-resnet-101',
                                  description='facebook/detr-resnet-101',
                                  tags=['pretrained', 'microsoft', 'ocr', 'facebook', 'huggingface'],
                                  dataset_id=None,
                                  scope='project',
                                  status='trained',
                                  labels=list(id2label.values()),
                                  configuration={'module_name': 'models.facebook.detr_resnet_101',
                                                 'id_to_label_map': id2label,
                                                 'label_to_id_map': {v: k for k, v in id2label.items()}},
                                  project_id=project.id,

                                  )
