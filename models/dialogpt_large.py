import dtlpy as dl
import torch
import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("[DialoGPT]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.top_k = configuration.get("top_k", 5)

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def compute_confidence(self, input_ids):
        logits = self.model(input_ids).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k)
        confidence_score = top_k_probs.sum().item()
        return confidence_score

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                chat_history_ids = torch.tensor([])
                questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content
                for i, question in enumerate(questions):
                    if question['mimetype'] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")
                        new_user_input_ids = self.tokenizer.encode(question["value"] + self.tokenizer.eos_token,
                                                                   return_tensors='pt')
                        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) \
                            if len(chat_history_ids) else new_user_input_ids
                        chat_history_ids = self.model.generate(bot_input_ids, max_new_tokens=1000, do_sample=True,
                                                               pad_token_id=self.tokenizer.eos_token_id,
                                                               top_k=self.top_k)
                        response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                                         skip_special_tokens=True)
                        logger.info("Response: {}".format(response))
                        item_annotations.add(annotation_definition=dl.FreeText(text=response),
                                             prompt_id=prompt_key,
                                             model_info={'name': "DialoGPT-Large",
                                                         'confidence': self.compute_confidence(new_user_input_ids)})
                    else:
                        logger.warning("Entry ignored. DialoGPT can only answer to text prompts.")
            annotations.append(item_annotations)
        return annotations


def model_creation(package: dl.Package):
    model = package.models.create(model_name='dialogpt-huggingface-test',
                                  description='dialogpt for chatting - HF',
                                  tags=['llm', 'pretrained', "hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'dialogpt.pt',
                                      "module_name": "models.dialogpt_large",
                                      'device': 'cpu'},
                                  project_id=package.project.id
                                  )
    return model
