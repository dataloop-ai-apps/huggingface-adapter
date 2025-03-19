
from dtlpy import BaseModelAdapter, PromptItem
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class HuggingAdapter(BaseModelAdapter):
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configuration.get('model_name', 'dslim/bert-base-NER'))
        self.model = AutoModelForTokenClassification.from_pretrained(self.configuration.get('model_name', 'dslim/bert-base-NER'))
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def prepare_item_func(self, item: PromptItem):
        return item.text  # Assuming the text to analyze is in the 'text' attribute of PromptItem

    def predict(self, item: PromptItem):
        text = self.prepare_item_func(item)
        ner_results = self.nlp(text)
        return ner_results
