import logging
import torch
from typing import List
from PIL import Image

import dtlpy as dl
from transformers import Owlv2Processor, Owlv2ForObjectDetection

logger = logging.getLogger("[OWLv2]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load OWLv2 model and processor.
        
        Args:
            local_path: Path to local model files (unused for pretrained models)
            **kwargs: Additional arguments
        """
        self.model_weights = self.configuration.get("model_weights", "google/owlv2-base-patch16")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = self.configuration.get("confidence_threshold", 0.2)
        self.label_source = self.configuration.get("label_source", "dataset_labels")
        self.custom_labels = self.configuration.get("custom_labels", [])
        
        logger.info(f"Loading OWLv2 model: {self.model_weights}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Label source: {self.label_source}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        valid_sources = ["dataset_labels", "custom"]
        if self.label_source not in valid_sources:
            raise ValueError(
                f"Invalid label_source: {self.label_source}. Must be one of {valid_sources}"
            )
        
        # Load model and processor
        self.processor = Owlv2Processor.from_pretrained(self.model_weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_weights)
        self.model.to(self.device)
        
        logger.info("OWLv2 model loaded successfully")

    def prepare_item_func(self, item: dl.Item):
        """Prepare item for prediction.
        
        Args:
            item: Dataloop item to prepare
            
        Returns:
            The item itself
        """
        if item.mimetype.startswith("image/"):
            return item
        else:
            raise ValueError(f"Unsupported item mimetype: {item.mimetype}")

    def _get_text_queries(self, item: dl.Item):
        """Get text queries based on configuration.
        
        Args:
            item: Dataloop item
            
        Returns:
            List of text queries for object detection
        """
        if self.label_source == "dataset_labels":
            try:
                labels = list(item.dataset.labels_flat_dict.keys())
                if not labels:
                    raise ValueError(
                        "label_source is set to 'dataset_labels' but item.dataset has no labels defined. "
                        "Please add labels to the dataset or change label_source to 'custom'."
                    )
                logger.info(f"Using dataset labels: {labels}")
                return labels
            except AttributeError:
                raise ValueError(
                    "label_source is set to 'dataset_labels' but item.dataset is not accessible. "
                    "Please ensure the item belongs to a dataset with labels or change label_source to 'custom'."
                )
        
        elif self.label_source == "custom":
            if not self.custom_labels:
                raise ValueError(
                    "label_source is set to 'custom' but custom_labels is empty. "
                    "Please provide a list of labels in the configuration."
                )
            logger.info(f"Using custom labels: {self.custom_labels}")
            return self.custom_labels
        else:
            raise ValueError(f"Invalid label_source: {self.label_source}")

    def _get_image_from_item(self, item: dl.Item):
        """Get PIL Image from item.
        
        Args:
            item: Dataloop item
            
        Returns:
            PIL Image object
        """
        buffer = item.download(save_locally=False)
        return Image.open(buffer).convert("RGB")

    def predict(self, batch, **kwargs):
        """Predict objects in images using text queries.
        
        Args:
            batch: List of prepared items from prepare_item_func
            **kwargs: Additional arguments
            
        Returns:
            List of AnnotationCollection objects
        """
        batch_annotations = []
        
        for item in batch:
            try:
                text_queries = self._get_text_queries(item)
                image = self._get_image_from_item(item)
                
                # Prepare inputs for OWLv2 (simultaneous processing of all queries)
                inputs = self.processor(
                    text=[[query for query in text_queries]],
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process results
                target_sizes = torch.Tensor([[image.size[1], image.size[0]]]).to(self.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.confidence_threshold
                )[0]
                
                item_annotations = dl.AnnotationCollection()
                
                for box, score, label_idx in zip(
                    results["boxes"],
                    results["scores"],
                    results["labels"]
                ):
                    score = score.item()
                    label_idx = label_idx.item()
                    box = box.tolist()
                    
                    label_text = text_queries[label_idx] if label_idx < len(text_queries) else f"object_{label_idx}"
                    
                    left, top, right, bottom = box
                    
                    # Clamp coordinates to image boundaries
                    img_width, img_height = image.size
                    left = max(0, min(left, img_width))
                    right = max(0, min(right, img_width))
                    top = max(0, min(top, img_height))
                    bottom = max(0, min(bottom, img_height))
                    
                    logger.info(f"Detected: {label_text} with confidence {score:.3f}")
                
                    item_annotations.add(
                        annotation_definition=dl.Box(
                            label=label_text,
                            top=top,
                            left=left,
                            bottom=bottom,
                            right=right
                        ),
                        model_info={
                            "name": self.model_entity.name,
                            "model_id": self.model_entity.id,
                            "confidence": round(score, 3),
                            "text_query": label_text
                        }
                    )
                
                batch_annotations.append(item_annotations)
                
            except Exception as e:
                logger.error(f"Error processing item {item.id if hasattr(item, 'id') else 'unknown'}: {str(e)}")
                batch_annotations.append(dl.AnnotationCollection())
        
        return batch_annotations

if __name__ == "__main__":
    dl.setenv('rc')
    item_id = ""
    model_id = ""
    
    item = dl.items.get(item_id=item_id)
    model_entity = dl.models.get(model_id=model_id)
    
    adapter = HuggingAdapter(model_entity=model_entity)
    adapter.predict_items(items=[item])
    
    print("Prediction complete!")