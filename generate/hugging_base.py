import torch
import base64
import logging
import dtlpy as dl

from io import BytesIO
from PIL import Image
from typing import List
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
        media_type = self.configuration.get("media_type", "multimodal")

        if media_type == "image":
            # For image-only models like DETR
            if 'image/' not in item.mimetype:
                raise ValueError(f"Item must be an image for {self.model_name}. Got mimetype: {item.mimetype}")
            return item
        elif media_type == "text":  
            # For text-only models like GPT 
            return item
        else:
            # For multimodal models like BLIP
            return dl.PromptItem.from_item(item)

    def predict(self, batch: List[dl.Item], **kwargs):
        """
        Process a batch of items and generate predictions.

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
                    "model_id": getattr(self, 'model_entity', {}).get('id', 1),
                },
            )

        return []

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
            image = Image.open(item.buffer).convert('RGB')
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
