import gc
import base64
import os
import re
import tempfile
from io import BytesIO
from typing import List

import dtlpy as dl
import logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger("[Cosmos-Reason2-8B]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        logger.info("-HHH- loding new version 0.1.83")

        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if self.hf_token is None:
            raise ValueError("Missing HUGGINGFACE_TOKEN environment variable")
        login(token=self.hf_token)

        hf_model_name = self.configuration.get("model_name", "nvidia/Cosmos-Reason2-8B")
        logger.info(f"Model name: {hf_model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Pixel limits control how many visual tokens the vision encoder produces.
        # Qwen3-VL default max_pixels is ~1.28 billion -- far too large for a T4.
        # 448*448=200704 → 256 visual tokens; 672*672=451584 → 576 visual tokens.
        min_pixels = self.configuration.get("min_pixels", 3136)
        max_pixels = self.configuration.get("max_pixels", 200704)
        self.max_image_size = self.configuration.get("max_image_size", 512)
        logger.info("Processor pixel limits: min_pixels=%d, max_pixels=%d, max_image_size=%d",
                     min_pixels, max_pixels, self.max_image_size)
        self.processor = AutoProcessor.from_pretrained(
            hf_model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

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
        self.max_new_tokens = self.configuration.get("max_new_tokens", 512)
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
        video_fps = self.configuration.get("video_fps", 2)
        max_video_frames = self.configuration.get("max_video_frames", 8)
        logger.info("video_fps config: %s, max_video_frames: %s", video_fps, max_video_frames)

        for prompt_item in batch:
            raw_messages = prompt_item.to_messages(model_name=self.configuration.get("model_name", "cosmos-reason2-8b"))
            logger.info("DEBUG raw_messages from prompt_item.to_messages(): %s", raw_messages)
            prompt_txt, image, video_path = self.reformat_messages(raw_messages)
            logger.info("DEBUG reformat_messages result: prompt_txt=%r, image_is_none=%s, video_path=%r",
                        prompt_txt, image is None, video_path)

            system_prompt = self.configuration.get("system_prompt", self.system_prompt)

            if image is not None:
                image = self._resize_image(image, self.max_image_size)

            user_content = []
            if image is not None:
                user_content.append({"type": "image", "image": image})
                logger.info("DEBUG user_content: added image (size=%s)", image.size)
            elif video_path is not None:
                user_content.append({
                    "type": "video",
                    "video": video_path,
                    "fps": video_fps,
                    "nframes": max_video_frames,
                })
                logger.info("DEBUG user_content: added video %s fps=%s nframes=%s", video_path, video_fps, max_video_frames)
            user_content.append({"type": "text", "text": prompt_txt})
            logger.info("DEBUG user_content types: %s", [c.get('type') for c in user_content])

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ]
            logger.info("DEBUG conversation structure: system + user with %d content blocks", len(user_content))

            chat_kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if video_path is not None:
                chat_kwargs["fps"] = video_fps
                logger.info("DEBUG chat_kwargs: added fps=%s for video", video_fps)

            logger.info("DEBUG calling processor.apply_chat_template...")
            inputs = self.processor.apply_chat_template(conversation, **chat_kwargs)
            logger.info("DEBUG apply_chat_template done, input keys: %s", list(inputs.keys()) if hasattr(inputs, 'keys') else type(inputs))
            inputs = inputs.to(self.model.device)

            if image is not None:
                image.close()
                del image
                image = None
            if video_path is not None:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            del conversation, user_content
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            input_len = inputs.input_ids.shape[-1] if hasattr(inputs, 'input_ids') else "unknown"
            logger.info("DEBUG input_ids length: %s, max_new_tokens: %s", input_len, self.max_new_tokens)
            if self.device.type == "cuda":
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                logger.info("DEBUG GPU mem before generate: allocated=%.2f GB, reserved=%.2f GB", alloc_gb, reserved_gb)
            logger.info("DEBUG calling model.generate with max_new_tokens=%s...", self.max_new_tokens)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            logger.info("DEBUG model.generate done, generated_ids shape: %s", generated_ids.shape)

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
    def _resize_image(image: Image.Image, max_size: int) -> Image.Image:
        """Downscale image so its longest side is at most max_size pixels."""
        w, h = image.size
        if max(w, h) <= max_size:
            return image
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info("Resizing image from %dx%d to %dx%d (max_size=%d)", w, h, new_w, new_h, max_size)
        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def get_last_prompt_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        raise ValueError("No message with role 'user' found")

    @staticmethod
    def _download_dataloop_video(video_url: str) -> str:
        """Download a video from a Dataloop URL and save to a temp file. Returns the file path."""
        logger.info("DEBUG _download_dataloop_video called with url: %s", video_url)
        item_id = video_url.split("items/")[1].split("/")[0]
        logger.info("DEBUG extracted item_id: %s", item_id)
        item = dl.items.get(item_id=item_id)
        logger.info("DEBUG item fetched: name=%s, mimetype=%s", item.name, item.mimetype)
        if not item.mimetype.startswith("video"):
            raise ValueError(f"Expected a video item, got {item.mimetype}")
        binaries = item.download(save_locally=False)
        logger.info("DEBUG downloaded binary size: %d bytes", len(binaries.getvalue()))
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(binaries.getvalue())
        tmp.close()
        logger.info("DEBUG saved video to temp file: %s", tmp.name)
        return tmp.name

    _VIDEO_MD_PATTERN = re.compile(r"\[video_url\]\(([^)]+)\)", re.IGNORECASE)

    @staticmethod
    def _extract_video_url_from_text(text: str):
        """Extract a [video_url](...) markdown link from text.

        Returns (clean_text, video_url) where video_url is the first Dataloop
        video link found, or None if there is none.
        """
        links = HuggingAdapter._VIDEO_MD_PATTERN.findall(text)
        clean_text = HuggingAdapter._VIDEO_MD_PATTERN.sub("", text).strip()
        for link in links:
            if "dataloop.ai/api/v1/items/" in link:
                return clean_text, link
        return clean_text, None

    @staticmethod
    def reformat_messages(messages):
        logger.info("DEBUG reformat_messages called with %d messages", len(messages))
        last_user_message = HuggingAdapter.get_last_prompt_message(messages)
        logger.info("DEBUG last_user_message: %s", last_user_message)

        prompt_txt = None
        image = None
        video_path = None

        content_list = last_user_message["content"]
        logger.info("DEBUG content_list has %d items", len(content_list))
        for i, content in enumerate(content_list):
            content_type = content.get("type", None)
            logger.info("DEBUG content[%d]: type=%r, keys=%s", i, content_type, list(content.keys()))
            if content_type is None:
                raise ValueError("Message content type not found")

            if content_type == "text":
                raw_text = content.get("text", "").strip()
                logger.info("DEBUG content[%d] text (first 200 chars): %r", i, raw_text[:200])

                clean_text, video_url = HuggingAdapter._extract_video_url_from_text(raw_text)
                if video_url is not None:
                    logger.info("DEBUG content[%d] found video markdown link in text: %s", i, video_url)
                    if video_path is not None:
                        raise ValueError("Multiple videos not supported")
                    video_path = HuggingAdapter._download_dataloop_video(video_url)
                    logger.info("DEBUG content[%d] video downloaded to: %s", i, video_path)

                if clean_text:
                    if prompt_txt is None:
                        prompt_txt = clean_text
                    else:
                        prompt_txt = f"{prompt_txt} {clean_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                logger.info("DEBUG content[%d] image_url present: %s, url starts with: %s",
                            i, image_url is not None, (image_url[:80] if image_url else "None"))
                if image_url is not None:
                    if image is not None:
                        raise ValueError("Multiple images not supported")
                    base64_str = content["image_url"]["url"].split("base64,")[1]
                    image_buffer = BytesIO(base64.b64decode(base64_str))
                    image = Image.open(image_buffer).convert("RGB")
                    logger.info("DEBUG content[%d] decoded image: size=%s", i, image.size)

            elif content_type == "video_url":
                if video_path is not None:
                    raise ValueError("Multiple videos not supported")
                url = content.get("video_url", {}).get("url", "")
                logger.info("DEBUG content[%d] video_url: %r", i, url)
                if "dataloop.ai/api/v1/items/" not in url:
                    raise ValueError(f"Unsupported video URL format (only Dataloop item URLs supported): {url[:100]}")
                video_path = HuggingAdapter._download_dataloop_video(url)
                logger.info("DEBUG content[%d] video downloaded to: %s", i, video_path)

            else:
                logger.error("DEBUG content[%d] UNSUPPORTED type: %r, full content: %s", i, content_type, content)
                raise ValueError(f"Unsupported content type: {content_type}")

        logger.info("DEBUG after parsing: prompt_txt=%r, image_is_none=%s, video_path=%r",
                    prompt_txt, image is None, video_path)

        if prompt_txt is None:
            if image is not None:
                prompt_txt = "What is in this image?"
            elif video_path is not None:
                prompt_txt = "Describe what happens in this video."

        if image is None and video_path is None:
            logger.error("DEBUG FAILURE: no image and no video found. Full messages dump: %s", messages)
            raise ValueError("No image or video found in messages.")

        logger.info("DEBUG reformat_messages returning: prompt_txt=%r, has_image=%s, video_path=%r",
                    prompt_txt, image is not None, video_path)
        return prompt_txt, image, video_path
