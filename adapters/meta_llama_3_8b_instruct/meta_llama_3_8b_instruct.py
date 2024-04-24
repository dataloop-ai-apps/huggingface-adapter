import dtlpy as dl
import torch
import json
import logging
from transformers import pipeline

logger = logging.getLogger("[LLaMa3]")


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", "meta-llama/Meta-Llama-3-8B-Instruct")
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
                                     "torch_dtype": torch_dtype,
                                     "cache_dir": configuration.get("cache_dir")
                                     },
                                 device_map=configuration.get("device", "cpu")
                                 )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        self.configuration = configuration
        self.configuration["model_entity"].artifacts.upload(configuration.get("cache_dir"), "*")
    
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
                        messages = [
                            {"role": "system", "content": self.configuration.get("system_prompt", "")},
                            {"role": "user", "content": question["value"]}
                            ]
                        template_prompt = self.pipeline.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                            )
                        logger.info(f"User: {question['value']}")
                        pipeline_output = self.pipeline(template_prompt,
                                                        max_new_tokens=self.configuration.get("max_new_tokens", 20),
                                                        eos_token_id=self.terminators,
                                                        do_sample=True,
                                                        temperature=self.configuration.get("temperature", 0.6),
                                                        top_p=self.configuration.get("top_p", 0.9))
                        response = pipeline_output[0]["generated_text"][len(template_prompt):]
                        logger.info("Response: {}".format(response))
                        item_annotations.add(annotation_definition=dl.FreeText(text=response), prompt_id=prompt_key,
                                             model_info={
                                                 "name": "LLaMa3-8b-Instruct",
                                                 "confidence": 1.0
                                             })
                    else:
                        logger.warning(f"LLaMa3 only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
