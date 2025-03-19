import dtlpy as dl
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List

logger = logging.getLogger("[BertBaseNER]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "bert-base-NER")
        self.device = self.configuration.get("device", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.model.to(self.device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def prepare_item_func(self, item: dl.Item):
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []
        for item in batch:
            if 'text/' in item.mimetype:
                text = item.download(save_locally=False).read().decode()
                ner_results = self.nlp(text)
                collection = dl.AnnotationCollection()
                for res in ner_results:
                    collection.add(
                        dl.Text(text_type='block', label=res['entity'], start=res['start'], end=res['end']),
                        model_info={'name': self.model_name, 'confidence': res['score']},
                    )
                batch_annotations.append(collection)
            elif 'json' in item.mimetype:
                item = dl.PromptItem.from_item(item)
                collection = dl.AnnotationCollection()
                for prompt in item.prompts:
                    if prompt.key == "text":
                        text = prompt.value
                        ner_results = self.nlp(text)
                        for res in ner_results:
                            collection.add(
                                dl.Text(text_type='block', label=res['entity'], start=res['start'], end=res['end']),
                                model_info={'name': self.model_name, 'confidence': res['score']},
                            )
                batch_annotations.append(collection)
        return batch_annotations
