import dtlpy as dl
import torch
import json
import logging
from open_lm.hf import *  # type: ignore # noqa: F403
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("[AppleDCLM7b]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name", "DCLM-Baseline-7B")
        self.tokenizer = AutoTokenizer.from_pretrained("apple/{}".format(self.model_name))
        self.model = AutoModelForCausalLM.from_pretrained("apple/{}".format(self.model_name))
        
        self.max_new_tokens = configuration.get("max_new_tokens", 50)
        self.top_p = configuration.get("top_p", 0.8)
        self.temperature = configuration.get("temperature", 0.7)
        self.do_sample = configuration.get("do_sample", True)
        self.repetition_penalty = configuration.get("repetition_penalty", 1.0)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def compute_confidence(self, input_ids):
        logits = self.model(input_ids).logits[:, -1, :]
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= self.top_p
        selected_probs = sorted_probs[mask]
        confidence_score = selected_probs.sum().item()
        return confidence_score

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                questions = (
                    list(prompt_content.values())
                    if isinstance(prompt_content, dict)
                    else prompt_content
                )

                for question in questions:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")
                        gen_kwargs = {
                            "max_new_tokens": self.max_new_tokens,
                            "top_p": self.top_p,
                            "temperature": self.temperature,
                            "do_sample": self.do_sample,
                            "repetition_penalty": self.repetition_penalty,
                        }
                        new_user_input_ids = self.tokenizer(
                            question["value"], return_tensors="pt"
                        )['input_ids']
                        generation_output = self.model.generate(
                            input_ids=new_user_input_ids, **gen_kwargs
                        )
                        response = self.tokenizer.decode(
                            generation_output[:, new_user_input_ids.shape[-1] + 1 :][0]
                        )
                        logger.info("Response: {}".format(response))
                        item_annotations.add(
                            annotation_definition=dl.FreeText(text=response),
                            prompt_id=prompt_key,
                            model_info={
                                "name": "DCLM_7B",
                                "confidence": self.compute_confidence(
                                    new_user_input_ids
                                ),
                            },
                        )
                    else:
                        logger.warning(
                            "DCLM only accepts text prompts, ignoring the current prompt."
                        )
            annotations.append(item_annotations)
        return annotations
