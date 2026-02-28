import gc
import base64
import os
from io import BytesIO
from typing import List

import dtlpy as dl
import logging
import torch
from PIL import Image
from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger("[Cosmos-Reason2-8B]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if self.hf_token is None:
            raise ValueError("Missing HUGGINGFACE_TOKEN environment variable")
        login(token=self.hf_token)

        hf_model_name = self.configuration.get("model_name", "nvidia/Cosmos-Reason2-8B")
        logger.info(f"Model name: {hf_model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(hf_model_name)

        # 4-bit NF4 quantization to reduce VRAM usage (~4-5 GB instead of ~16 GB),
        # required to fit the 8B model on a T4 (16 GB) GPU.
        # Use float16 compute dtype -- T4 (sm_75) does not natively support bfloat16.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_model_name,
            quantization_config=quantization_config,
            device_map="cuda:0",
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
        )
        self.max_new_tokens = self.configuration.get("max_new_tokens", 4096)
        self.system_prompt = self.configuration.get(
            "system_prompt",
            "You are a helpful assistant. Describe what you see in the image accurately and answer questions related to the visual content.",
        )

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.PromptItem], **kwargs):
        logger.info("Predicting on device: %s", self.device)
        if self.device.type == "cuda":
            logger.info(
                "Available GPU memory: %.2f GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        for prompt_item in batch:
            prompt_txt, image = HuggingAdapter.reformat_messages(
                prompt_item.to_messages(model_name=self.configuration.get("model_name", "cosmos-reason2-8b"))
            )

            system_prompt = self.configuration.get("system_prompt", self.system_prompt)
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_txt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            del inputs, generated_ids, generated_ids_trimmed

            logger.info("Response: %s", response)
            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": response}],
                },
                model_info={
                    "name": self.configuration.get("model_name", "cosmos-reason2-8b"),
                    "confidence": 1.0,
                    "model_id": self.model_entity.id,
                },
            )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        return []

    @staticmethod
    def get_last_prompt_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")

    @staticmethod
    def reformat_messages(messages):
        last_user_message = HuggingAdapter.get_last_prompt_message(messages)

        prompt_txt = None
        image = None

        for content in last_user_message["content"]:
            content_type = content.get("type", None)
            if content_type is None:
                raise ValueError("Message content type not found")

            if content_type == "text":
                new_text = content.get("text", "").strip()
                if new_text:
                    if prompt_txt is None:
                        prompt_txt = new_text
                    else:
                        prompt_txt = f"{prompt_txt} {new_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url is not None:
                    if image is not None:
                        raise ValueError("Multiple images not supported")
                    base64_str = content["image_url"]["url"].split("base64,")[1]
                    image_buffer = BytesIO(base64.b64decode(base64_str))
                    image = Image.open(image_buffer).convert("RGB")
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        if prompt_txt is None:
            prompt_txt = "What is in this image?"

        if image is None:
            raise ValueError("No image found in messages.")

        return prompt_txt, image
