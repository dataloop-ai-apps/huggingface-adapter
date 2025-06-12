import torch
import base64
import logging
import dtlpy as dl
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger("[HuggingFaceAdapter]")


class HuggingBase(dl.BaseModelAdapter, ABC):
    """
    Base class for Hugging Face model adapters with common functionality.
    Implements shared methods from different model adapters.
    """

    def load(self, local_path, **kwargs):
        """
        Initialize the model and processor.
        This method sets common configuration parameters but requires
        implementation-specific model loading.

        Args:
            local_path: Path to the model
            **kwargs: Additional arguments
        """
        # Load model and processor (to be implemented by child classes)
        self.load_model_and_processor()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conditioning = self.configuration.get("conditioning", False)
        self.model.to(self.device)

    @abstractmethod
    def load_model_and_processor(self):
        """
        Load the specific model and processor.
        This method must be implemented by child classes.
        """
        self.model_name = self.configuration.get("model_name")
        self.processor = None
        self.model = None

        return

    def prepare_item_func(self, item: dl.Item):
        """
        Prepare an item for inference by checking its mimetype and converting to appropriate format.

        Args:
            item: Input item

        Returns:
            Union[dl.Item, dl.PromptItem]: Prepared item in appropriate format
        """
        # Get model type from configuration
        media_type = self.configuration.get("media_type", None)

        if media_type is None:
            raise ValueError(f"Media type not specified for {self.model_name}")
        elif media_type == "image":
            # For image-only models like DETR
            if "image/" not in item.mimetype:
                raise ValueError(f"Item must be an image for {self.model_name}. Got mimetype: {item.mimetype}")
            item = item
        elif media_type == "text":
            # For text-only models like GPT
            item = item
        else:
            # For multimodal models like BLIP
            item = dl.PromptItem.from_item(item)
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of items and generate predictions.
        Supports both regular image items and prompt items.

        Args:
            batch: List of Items or PromptItems to process
            **kwargs: Additional arguments

        Returns:
            List: List of annotations for regular items, empty list for prompt items
        """
        results = []

        for item in batch:
            if isinstance(item, dl.PromptItem):
                # Handle prompt items (multimodal)
                self._process_prompt_item(item)
            else:
                # Handle regular image items
                annotation = self._process_regular_item(item)
                if annotation:
                    results.append(annotation)

        return results

    def _process_prompt_item(self, prompt_item: dl.PromptItem):
        """
        Process a prompt item for multimodal models.

        Args:
            prompt_item: PromptItem to process
        """
        # Get the formatted prompt text and image buffer
        prompt_txt, image_buffer = self.reformat_messages(prompt_item.to_messages(model_name=self.model_name))

        # Process the image and text with the model
        encoding = self._prepare_encoding(dl.PromptItem(image_buffer, prompt_txt))

        # Generate output
        output = self.model.generate(**encoding, max_new_tokens=50)

        # Decode the output
        response = self.processor.decode(output[0], skip_special_tokens=True).strip()

        # Log the response
        logger.debug(f"Response: {response}")

        # Add the response to the prompt item
        prompt_item.add(
            message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
            model_info={
                "name": self.model_name,
                "confidence": 1.0,
                "model_id": getattr(self, "model_entity", {}).get("id", 1),
            },
        )

        return []

    def _process_regular_item(self, item: dl.Item):
        """
        Process a regular image item.

        Args:
            item: Item to process

        Returns:
            dl.Annotation: Created annotation or None if processing failed
        """
        result = None
        try:
            # Prepare the item
            prepared_item = self.prepare_item_func(item)
            encoding = self._prepare_encoding(prepared_item)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**encoding)

            # Post-process the outputs
            result = self.post_process(item, outputs, encoding)

        except Exception as e:
            logger.error(f"Error processing item {item.id}: {str(e)}")
            result = None

        return result

    def post_process(self, item: dl.Item, outputs: Any, encoding: Any) -> dl.Annotation:
        """
        Post-process model outputs to create annotations.
        This method can be overridden by subclasses to implement custom post-processing.

        Args:
            item: Original item
            outputs: Model outputs
            encoding: Model input encoding

        Returns:
            dl.Annotation: Processed annotation or None if no post-processing needed
        """
        result = None
        # Get model type from configuration
        model_type = self.configuration.get("model_type", "classification")

        # Process based on model type
        if model_type == "classification":
            result = self._create_classification_annotation(item, outputs)
        elif model_type == "detection":
            result = self._create_detection_annotation(item, outputs)
        elif model_type == "segmentation":
            result = self._create_segmentation_annotation(item, outputs)
        else:
            logger.warning(f"Unsupported model type: {model_type}")

        return result

    def _create_classification_annotation(self, item: dl.Item, outputs) -> dl.Annotation:
        """
        Create a classification annotation from model outputs.

        Args:
            item: Original item
            outputs: Model outputs

        Returns:
            dl.Annotation: Classification annotation
        """
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

        # Get label from processor or configuration
        label = self.processor.id2label[pred_idx] if hasattr(self.processor, 'id2label') else str(pred_idx)

        # Create classification annotation
        annotation = dl.AnnotationCollection()
        annotation.add(
            annotation_definition=dl.Classification(label=label),
            model_info={'name': self.model_name, 'model_id': self.model_entity.id, 'confidence': float(confidence)},
        )

        return annotation

    def _create_detection_annotation(self, item: dl.Item, outputs) -> dl.Annotation:
        """
        Create a detection annotation from model outputs.

        Args:
            item: Original item
            outputs: Model outputs

        Returns:
            dl.Annotation: Detection annotation
        """
        # Get predictions
        pred_boxes = outputs.pred_boxes if hasattr(outputs, 'pred_boxes') else outputs[0]
        pred_scores = outputs.scores if hasattr(outputs, 'scores') else outputs[1]
        pred_labels = outputs.pred_labels if hasattr(outputs, 'pred_labels') else outputs[2]

        # Create detection annotation
        annotation = dl.AnnotationCollection()

        for box, score, label_idx in zip(pred_boxes, pred_scores, pred_labels):
            if score > self.configuration.get("confidence_threshold", 0.5):
                label = (
                    self.processor.id2label[label_idx.item()]
                    if hasattr(self.processor, 'id2label')
                    else str(label_idx.item())
                )

                # Convert box coordinates to Dataloop format
                box = box.tolist()
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

                annotation.add(
                    annotation_definition=dl.Box(
                        label=label,
                        left=max(x, 0),
                        top=max(y, 0),
                        right=min(x + w, item.width),
                        bottom=min(y + h, item.height),
                    ),
                    model_info={
                        'name': self.model_name,
                        'model_id': self.model_entity.id,
                        'confidence': float(score.item()),
                    },
                )

        return annotation

    def _create_segmentation_annotation(self, item: dl.Item, outputs) -> dl.Annotation:
        """
        Create a segmentation annotation from model outputs.

        Args:
            item: Original item
            outputs: Model outputs

        Returns:
            dl.Annotation: Segmentation annotation
        """
        masks = outputs.pred_masks if hasattr(outputs, 'pred_masks') else outputs[0]
        scores = outputs.scores if hasattr(outputs, 'scores') else outputs[1]
        labels = outputs.pred_labels if hasattr(outputs, 'pred_labels') else outputs[2]

        # Create segmentation annotation
        annotation = dl.AnnotationCollection()

        for mask, score, label_idx in zip(masks, scores, labels):
            if score > self.configuration.get("confidence_threshold", 0.5):
                label = (
                    self.processor.id2label[label_idx.item()]
                    if hasattr(self.processor, 'id2label')
                    else str(label_idx.item())
                )

                # Convert mask to polygon
                mask = mask.cpu().numpy()
                contours = self._mask_to_polygon(mask)

                for contour in contours:
                    annotation.add(
                        annotation_definition=dl.Polygon(label=label, geo=contour),
                        model_info={
                            'name': self.model_name,
                            'model_id': self.model_entity.id,
                            'confidence': float(score.item()),
                        },
                    )

        return annotation

    def _mask_to_polygon(self, mask):
        """
        Convert binary mask to polygon contours.

        Args:
            mask: Binary mask array

        Returns:
            List: List of polygon contours
        """
        import cv2
        import numpy as np

        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to polygons
        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert to list of points
            polygon = approx.reshape(-1, 2).tolist()
            if len(polygon) > 2:  # Only add if it's a valid polygon
                polygons.append(polygon)

        return polygons

    def _prepare_encoding(self, item):
        """
        Prepare the encoding for the model based on the item content.
        This can be overridden by child classes for specific implementation.

        Args:
            item: Input item containing image and optional text

        Returns:
            Dict: Encoding for the model
        """
        args = []

        if "image" in item.mimetype:
            buffer = item.download(save_locally=False)
            image = Image.open(buffer).convert("RGB")
            args.append(image)
        elif "text" in item.mimetype:
            args.append(item.text)
        else:
            raise ValueError("Item must be an image or text")

        processor_fn = self.processor
        if isinstance(self.processor, dict):
            processor_fn = self.processor.get(args[0].__class__.__name__, None)
            if processor_fn is None:
                raise ValueError(f"No processor found for type {args[0].__class__.__name__}")

        return processor_fn(*args, return_tensors="pt").to(self.device)

    @staticmethod
    def get_last_prompt_message(messages):
        """
        Get the last message with role 'user'.

        Args:
            messages: List of messages

        Returns:
            Dict: Last user message
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")

    @staticmethod
    def reformat_messages(messages):
        """
        Reformat messages to extract the prompt text and image buffer.

        Args:
            messages: List of messages

        Returns:
            Tuple: (prompt_txt, image_buffer)

        """
        # Get the last user message
        last_user_message = HuggingBase.get_last_prompt_message(messages)

        prompt_txt = None
        image_buffer = None

        # Process each content in the message
        for content in last_user_message["content"]:
            content_type = content.get("type")

            if content_type == "text":
                # Concatenate multiple text contents with space
                new_text = content.get("text", "").strip()
                if new_text:
                    if prompt_txt is None:
                        prompt_txt = new_text
                    else:
                        prompt_txt = f"{prompt_txt} {new_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url:
                    if image_buffer:  # already found an image
                        logger.warning("Multiple images not supported, using only the first one")
                    else:
                        base64_str = content["image_url"]["url"].split("base64,")[1]
                        image_buffer = BytesIO(base64.b64decode(base64_str))
            else:
                logger.warning(f"Unsupported content type: {content_type}")

        # Set default prompt if none provided
        if prompt_txt is None:
            prompt_txt = "What is in this image?"

        # Format the prompt
        prompt_txt = f"Question: {prompt_txt} Answer:"

        # Check if an image was found
        if image_buffer is None:
            raise ValueError("No image found in messages.")

        return prompt_txt, image_buffer
