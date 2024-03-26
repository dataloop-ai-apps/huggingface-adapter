import dtlpy as dl
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("[amazon-review-sentiment-analysis]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
        self.model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

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
                        new_user_input_ids = self.tokenizer(question['value'], return_tensors="pt")
                        with torch.no_grad():
                            logits = self.model(**new_user_input_ids).logits
                        predicted_class_id = logits.argmax().item()
                        response = self.model.config.id2label[predicted_class_id]
                        logger.info("Response: {}".format(response))
                        item_annotations.add(annotation_definition=dl.FreeText(text=response), prompt_id=prompt_key,
                                             model_info={
                                                 "name": "Amazon Review Sentiment Analysis",
                                                 "confidence": round(logits.softmax(dim=1)[0, predicted_class_id].item(), 3)
                        })
                    else:
                        logger.warning(f"Model only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
