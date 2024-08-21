import dtlpy as dl
import torch
import json
import logging
from transformers import LlamaTokenizer, LlamaForCausalLM

logger = logging.getLogger("[Open Llama 3b]")


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", 'openlm-research/open_llama_3b')
        torch_dtype = configuration.get("torch_dtype")
        if torch_dtype == 'fp32':
            torch_dtype = torch.float32
        elif torch_dtype == 'fp16':
            torch_dtype = torch.float16
        elif torch_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32 if configuration.get("device", "cpu") else None
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path,
                                                      torch_dtype=torch_dtype,
                                                      device_map='auto')
        self.top_k = configuration.get("top_k", 5)
    
    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def compute_confidence(self, input_ids):
        logits = self.model(input_ids).logits[:, -1, :]
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k)
        confidence_score = top_k_probs.sum().item()
        return confidence_score

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        model_device = self.model.device
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

                for question in questions:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")
                        new_user_input_ids = self.tokenizer(question['value'],
                                                            return_tensors='pt').input_ids.to(model_device)
                        generation_output = self.model.generate(input_ids=new_user_input_ids, max_length=100)
                        response = self.tokenizer.decode(generation_output[:, new_user_input_ids.shape[-1] + 1:][0])
                        logger.info("Response: {}".format(response))
                        item_annotations.add(
                            annotation_definition=dl.FreeText(text=response),
                            prompt_id=prompt_key,
                            model_info={
                                "name": "OpenLlama",
                                "confidence": self.compute_confidence(new_user_input_ids)
                            }
                        )
                    else:
                        logger.warning(f"OpenLlama only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
