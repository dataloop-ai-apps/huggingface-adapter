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

    def prepare_item_func(self, item: dl.Item):
        image = item.download(save_locally=False, to_array=True)
        return image
