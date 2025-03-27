import torch
import base64
import logging
import dtlpy as dl

from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple
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
        # Get model name and device from configuration
        self.model_name = self.configuration.get("model_name")
        self.device = self.configuration.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.media_type = self.configuration.get("media_type", "multimodal")  # text, image, or multimodal
        
        # Load specific model and processor
        self.load_model_and_processor()
        
        # Move model to device
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(self.device)

    @abstractmethod
    def load_model_and_processor(self):
        """
        Load the specific model and processor.
        This method must be implemented by child classes.
        """
        pass

    def prepare_item_func(self, item: dl.Item):
        """
        Prepare an item for inference by checking its mimetype and converting to appropriate format.

        Args:
            item: Input item

        Returns:
            Union[dl.Item, dl.PromptItem]: Prepared item in appropriate format
        """
        # Get model type from configuration
        media_type = self.media_type

        if media_type == "image":
            # For image-only models like DETR or image captioning
            if 'image/' not in item.mimetype:
                raise ValueError(f"Item must be an image for {self.model_name}. Got mimetype: {item.mimetype}")
            return item
        elif media_type == "text":  
            # For text-only models like summarization
            if "text" not in item.mimetype and "json" not in item.mimetype:
                raise ValueError(f"Item must be text for {self.model_name}. Got mimetype: {item.mimetype}")
            return item
        else:
            # For multimodal models like BLIP
            return dl.PromptItem.from_item(item)

    def predict(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of items and generate predictions.
        This implementation handles prompt-based models. Override for other model types.

        Args:
            batch: List of items to process
            **kwargs: Additional arguments

        Returns:
            List: Results (may be empty if results are added directly to items)
        """
        if self.media_type == "multimodal":
            return self._predict_multimodal(batch, **kwargs)
        elif self.media_type == "text":
            return self._predict_text(batch, **kwargs)
        elif self.media_type == "image":
            return self._predict_image(batch, **kwargs)
        else:
            raise ValueError(f"Unsupported media type: {self.media_type}")

    def _predict_multimodal(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of items for multimodal models (e.g., VQA).
        
        Args:
            batch: List of PromptItems to process
            **kwargs: Additional arguments
            
        Returns:
            List: Empty list (results are added to prompt items)
        """
        for prompt_item in batch:
            # Get the formatted prompt text and image buffer
            prompt_txt, image_buffer = self.reformat_messages(prompt_item.to_messages(model_name=self.model_name))

            # Process the image and text with the model
            encoding = self._prepare_encoding(prompt_txt, image_buffer)

            # Generate output
            output = self._generate_output(encoding, **kwargs)

            # Decode the output
            response = self._decode_output(output)

            # Log the response
            logger.debug(f"Response: {response}")

            # Add the response to the prompt item
            prompt_item.add(
                message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
                model_info={
                    "name": self.model_name,
                    "confidence": 1.0,
                    "model_id": getattr(self, 'model_entity', {}).get('id', 1),
                },
            )

        return []
    
    def _predict_text(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of text items (e.g., for summarization).
        Override in child classes as needed.
        
        Args:
            batch: List of text items to process
            **kwargs: Additional arguments
            
        Returns:
            List: Results
        """
        raise NotImplementedError("Text prediction not implemented in base class")
    
    def _predict_image(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of image items (e.g., for object detection).
        Override in child classes as needed.
        
        Args:
            batch: List of image items to process
            **kwargs: Additional arguments
            
        Returns:
            List: Results (e.g., annotations)
        """
        raise NotImplementedError("Image prediction not implemented in base class")

    def _prepare_encoding(self, prompt_txt: Optional[str], image_buffer: Optional[BytesIO]) -> Dict[str, Any]:
        """
        Prepare the encoding for the model based on text and/or image.
        Override in child classes for specific implementation.

        Args:
            prompt_txt: Text prompt (if applicable)
            image_buffer: Image buffer (if applicable)

        Returns:
            Dict: Encoding for the model
        """
        encoding = {}
        
        if image_buffer is not None and hasattr(self, 'processor'):
            image = Image.open(image_buffer).convert('RGB')
            if prompt_txt is not None:
                # For VQA models that take both image and text
                encoding = self.processor(images=image, text=prompt_txt, return_tensors="pt").to(self.device)
            else:
                # For image-only models
                encoding = self.processor(images=image, return_tensors="pt").to(self.device)
        elif prompt_txt is not None and hasattr(self, 'tokenizer'):
            # For text-only models
            encoding = self.tokenizer(prompt_txt, return_tensors="pt", truncation=True).to(self.device)
        
        return encoding

    def _generate_output(self, encoding: Dict[str, Any], **kwargs) -> Any:
        """
        Generate output from the model using the prepared encoding.
        Override in child classes for specific implementation.

        Args:
            encoding: Prepared encoding
            **kwargs: Additional generation arguments

        Returns:
            Any: Model output
        """
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 50),
            "num_beams": kwargs.get("num_beams", 4),
        }
        return self.model.generate(**encoding, **generation_kwargs)

    def _decode_output(self, output: Any) -> str:
        """
        Decode the output from the model.
        Override in child classes for specific implementation.

        Args:
            output: Model output

        Returns:
            str: Decoded output
        """
        if hasattr(self, 'processor') and hasattr(self.processor, 'decode'):
            return self.processor.decode(output[0], skip_special_tokens=True).strip()
        elif hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        elif hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'batch_decode'):
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        else:
            raise NotImplementedError("No decode method available")

    @staticmethod
    def get_last_prompt_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    def reformat_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[BytesIO]]:
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
        for content in last_user_message.get("content", []):
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
                if image_url and "base64," in image_url:
                    if image_buffer:  # already found an image
                        logger.warning("Multiple images not supported, using only the first one")
                    else:
                        base64_str = image_url.split("base64,")[1]
                        image_buffer = BytesIO(base64.b64decode(base64_str))
            else:
                logger.warning(f"Unsupported content type: {content_type}")

        return prompt_txt, image_buffer

    def extract_text_from_item(self, item: dl.Item) -> str:
        """
        Extract text from a text item.

        Args:
            item: Input text item

        Returns:
            str: Extracted text
        """
        if "text" in item.mimetype:
            text = item.download(save_locally=False).read().decode("utf-8")
        else:
            # Handle JSON prompt item
            prompt_item = dl.PromptItem.from_item(item)
            messages = prompt_item.to_messages(model_name=self.model_name)
            text = None
            for message in messages:
                if isinstance(message.get("content"), list):
                    for content in message.get("content", []):
                        if content.get("type") == "text" or "text" in content:
                            text = content.get("text", content.get("value", ""))
                            if text:
                                break
            
            if not text:
                logger.error(f"No text found in item {item.id}")
                text = ""
                
        return text