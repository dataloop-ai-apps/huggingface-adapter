import dtlpy as dl
import torch
import json
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, DefaultDataCollator
from datasets import load_dataset


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", 'openlm-research/open_llama_3b')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
        self.top_k = configuration.get("top_k", 5)
        self.configuration = configuration
    
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
        for item in batch:
            prompts = item["prompts"]
            item_annotations = []
            for prompt_key, prompt_content in prompts.items():
                for question in prompt_content.values():
                    print(f"User: {question['value']}")
                    new_user_input_ids = self.tokenizer(question['value'], return_tensors='pt').input_ids
                    generation_output = self.model.generate(input_ids=new_user_input_ids, max_length=100)
                    response = self.tokenizer.decode(generation_output[:, new_user_input_ids.shape[-1] + 1:][0])
                    print("Response: {}".format(response))
                    item_annotations.append({
                        "type": "text",
                        "label": "q",
                        "coordinates": response,
                        "metadata": {
                            "system": {"promptId": prompt_key},
                            "user": {
                                "annotation_type": "prediction",
                                "model": {
                                    "name": "OpenLLaMa",
                                    "confidence": self.compute_confidence(new_user_input_ids)
                                    }
                                }}
                        })
            annotations.append(item_annotations)
            return annotations

    @staticmethod
    def convert_from_dtlpy(data_path, **kwargs):
        for subset in ["train", "validation"]:
            os.makedirs(os.path.join(data_path, subset,'texts'), exist_ok=True)
            for json_file in os.listdir(os.path.join(data_path, subset)):
                with open(json_file, 'r') as jf:
                    prompts = json.load(jf)["prompts"]
                    for prompt_key, prompt_content in prompts.items():
                        for i, question in enumerate(prompt_content.values()):
                            with open(os.path.join(data_path, subset, 'texts', f"{prompt_key}_{i}.txt"), 'w') \
                                     as text_file:
                                text_file.write(question['value'])

    def create_training_args(self, data_path, output_path, **kwargs):
        return TrainingArguments(
                output_dir=output_path,
                evaluation_strategy=self.configuration.get("evaluation_strategy", "epoch"),
                learning_rate=self.configuration.get("learning_rate", 2e-5),
                per_device_train_batch_size=self.configuration.get("train_batch_size", 16),
                per_device_eval_batch_size=self.configuration.get("eval_batch_size", 16),
                num_train_epochs=self.configuration.get("epochs", 3),
                weight_decay=self.configuration.get("weight_decay", 0.01),
                push_to_hub=self.configuration.get("push_to_hub", False)
            )

    def create_datasets(self, data_path):
        train_path = os.path.join(data_path, 'train', 'texts')
        evaluate_path = os.path.join(data_path, 'validation', 'texts')
        dataset = load_dataset("text",
                               data_files={'train': os.listdir(train_path),
                                           'test': os.listdir(evaluate_path)},
                               sample_by='paragraph')

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset['train'], tokenized_dataset['test']

    def select_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def create_data_collator():
        return DefaultDataCollator()


def model_creation(package: dl.Package):

    model = package.models.create(model_name='openllama-huggingface',
                                  description='openllama for chatting - HF',
                                  tags=['llm', 'pretrained', "hugging-face"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={
                                      'weights_filename': 'openllama.pt',
                                      'model_path': 'openlm-research/open_llama_3b',
                                      "module_name": "models.open_llama",
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model
