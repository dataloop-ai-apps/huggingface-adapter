import dtlpy as dl
import os
import torch
import json
import logging
from transformers import pipeline, BitsAndBytesConfig

logger = logging.getLogger("[LLaMa3]")


class HuggingAdapter:
    def __init__(self, configuration):
        access_token = os.getenv(configuration.get("hf_access_token", "HUGGINGFACEHUB_API_KEY"))
        model_path = configuration.get("model_path", "meta-llama/Meta-Llama-3-8B-Instruct")
        torch_dtype = configuration.get("torch_dtype", "fp16")
        model_args = {"low_cpu_mem_usage": True}
        if torch_dtype == 'fp32':
            torch_dtype = torch.float32
            model_args["torch_dtype"] = torch_dtype
        elif torch_dtype == 'fp16':
            torch_dtype = torch.float16
            model_args["torch_dtype"] = torch_dtype
        elif torch_dtype == 'bf16':
            torch_dtype = torch.bfloat16
            model_args["torch_dtype"] = torch_dtype
        elif torch_dtype == "4bits":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                )
            model_args["quantization_config"] = bnb_config
        else:
            torch_dtype = torch.float32 if configuration.get("device", "cpu") else None
            model_args["torch_dtype"] = torch_dtype
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs=model_args,
            device_map=configuration.get("device", "cpu"),
            token=access_token
            )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
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
                        pipeline_output = self.pipeline(
                            template_prompt,
                            max_new_tokens=self.configuration.get("max_new_tokens", 20),
                            eos_token_id=self.terminators,
                            do_sample=True,
                            temperature=self.configuration.get("temperature", 0.6),
                            top_p=self.configuration.get("top_p", 0.9)
                            )
                        response = pipeline_output[0]["generated_text"][len(template_prompt):]
                        logger.info("Response: {}".format(response))
                        item_annotations.add(
                            annotation_definition=dl.FreeText(text=response), prompt_id=prompt_key,
                            model_info={
                                "name": "LLaMa3-8b-Instruct",
                                "confidence": 1.0
                                }
                            )
                    else:
                        logger.warning(f"LLaMa3 only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
