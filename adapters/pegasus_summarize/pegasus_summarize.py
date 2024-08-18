import dtlpy as dl
import logging
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import json

logger = logging.getLogger("[pegasus-summarize]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.model_name = configuration.get("model_name")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

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
                for question in prompt_content:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        print(f"User: {question['value']}")
                        tensor = self.tokenizer(question['value'], truncation=True, padding="longest", return_tensors="pt").to(self.device)
                        summary = self.model.generate(**tensor)
                        summary_text = self.tokenizer.batch_decode(summary, skip_special_tokens=True)[0]
                        item_annotations.add(
                            annotation_definition=dl.FreeText(text=summary_text),
                            prompt_id=prompt_key,
                            model_info={
                                "name": "Pegasus",
                                "confidence": 1.0
                            }
                        )
                    else:
                        logger.warning(f"Pegasus only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
