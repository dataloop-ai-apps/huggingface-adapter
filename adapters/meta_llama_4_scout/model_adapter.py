import logging
import dtlpy as dl
from typing import List
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

logger = logging.getLogger("[LLAMA-4]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name, attn_implementation="flex_attention", device_map="auto", torch_dtype=torch.bfloat16
        )

    def prepare_item_func(self, item: dl.Item):
        if not isinstance(item, dl.PromptItem):
            raise ValueError("Item must be a PromptItem for multimodal processing.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []

        for item in batch:
            try:
                prompt_item = dl.PromptItem.from_item(item)
                messages = prompt_item.to_messages(model_name=self.model_name)

                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)

                outputs = self.model.generate(**inputs, max_new_tokens=256)

                response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[0]
                prompt_item.add(
                    message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
                    model_info={"name": self.model_name, "confidence": 1.0, "model_id": 1},
                )
                batch_annotations.append(prompt_item)

            except Exception as e:
                logger.error(f"Error processing item {item.id}: {str(e)}")

        return batch_annotations
