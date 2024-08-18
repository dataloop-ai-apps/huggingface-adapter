from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import dtlpy as dl
import logging

logger = logging.getLogger("[BertBaseNER]")


class HuggingAdapter:
    def __init__(self, configuration):
        self.device = configuration.get("device")

        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.model.to(self.device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def prepare_item_func(self, item: dl.Item):
        buffer = item.download(save_locally=False)
        text = buffer.read().decode()
        return text

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, model_entity, batch, **kwargs):
        batch_annotations = list()
        for txt in batch:
            ner_results = self.nlp(txt)
            collection = dl.AnnotationCollection()
            for res in ner_results:
                collection.add(
                    annotation_definition=dl.Text(
                        text_type='block',
                        label=res['entity'],
                        start=res['start'],
                        end=res['end']
                    ),
                    model_info={
                        'name': model_entity.name,
                        'confidence': res['score']
                    }
                )
            batch_annotations.append(collection)
        return batch_annotations
