import dtlpy as dl
import torch
import json
import logging
from transformers import pipeline

logger = logging.getLogger("[LLaMa3]")


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", "meta-llama/Meta-Llama-3-8B")
        torch_dtype = configuration.get("torch_dtype", "fp16")
        if torch_dtype == 'fp32':
            torch_dtype = torch.float32
        elif torch_dtype == 'fp16':
            torch_dtype = torch.float16
        elif torch_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32 if configuration.get("device", "cpu") else None
        self.pipeline = pipeline("text-generation",
                                 model=model_path,
                                 model_kwargs={
                                     "low_cpu_mem_usage": True,
                                     "torch_dtype": torch_dtype
                                     },
                                 device_map=configuration.get("device", "cpu"))
        self.configuration = configuration
    
    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

                for question in questions:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")
                        pipeline_output = self.pipeline(question["value"],
                                                        max_new_tokens=self.configuration.get("max_new_tokens", 20))
                        response = pipeline_output[0]["generated_text"][len(question['value']):]
                        logger.info("Response: {}".format(response))
                        item_annotations.add(annotation_definition=dl.FreeText(text=response), prompt_id=prompt_key,
                                             model_info={
                                                 "name": "LLaMa3",
                                                 "confidence": 1.0
                                             })
                    else:
                        logger.warning(f"LLaMa3 only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
