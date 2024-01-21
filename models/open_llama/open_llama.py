import dtlpy as dl
import torch
import json
from transformers import LlamaTokenizer, LlamaForCausalLM


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", 'openlm-research/open_llama_3b')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
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

    def predict(self, batch, **kwargs):
        annotations = []
        model_device = self.model.device
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                for question in prompt_content:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        print(f"User: {question['value']}")
                        new_user_input_ids = self.tokenizer(question['value'],
                                                            return_tensors='pt').input_ids.to(model_device)
                        generation_output = self.model.generate(input_ids=new_user_input_ids, max_length=100)
                        response = self.tokenizer.decode(generation_output[:, new_user_input_ids.shape[-1] + 1:][0])
                        print("Response: {}".format(response))
                        item_annotations.add(annotation_definition=dl.FreeText(text=response), prompt_id=prompt_key,
                                             model_info={
                                                 "name": "OpenLlama",
                                                 "confidence": self.compute_confidence(new_user_input_ids)
                                             })
                    else:
                        print(f"OpenLlama only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations


def model_creation(package: dl.Package):
    model = package.models.create(model_name='openllama-huggingface',
                                  description='openllama for chatting - HF',
                                  tags=['llm', 'pretrained', "hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': 'openllama.pt',
                                      'model_path': 'openlm-research/open_llama_3b',
                                      "module_name": "models.open_llama",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
