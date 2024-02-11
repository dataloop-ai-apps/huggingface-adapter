from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import dtlpy as dl


class HuggingAdapter:
    def __init__(self, configuration=None):
        self.configuration = configuration if configuration else {}
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def prepare_item_func(self, item: dl.Item):
        buffer = item.download(save_locally=False)
        text = buffer.read().decode()
        return text

    def predict(self, model_entity, batch, **kwargs):
        batch_annotations = list()
        for txt in batch:
            ner_results = self.nlp(txt)
            collection = dl.AnnotationCollection()
            for res in ner_results:
                collection.add(dl.Text(text_type='block',
                                       label=res['entity'],
                                       start=res['start'],
                                       end=res['end']),
                               model_info={'name': model_entity.name,
                                           'confidence': res['score']})
            batch_annotations.append(collection)
        return batch_annotations


def create_model_entity(package: dl.Package) -> dl.Model:
    hugging = HuggingAdapter()
    id2label = hugging.model.config.id2label
    model = package.models.create(model_name='dslim/bert-base-NER',
                                  description='dslim/bert-base-NER',
                                  tags=['pretrained', 'ner', 'huggingface'],
                                  dataset_id=None,
                                  scope='project',
                                  status='trained',
                                  labels=list(id2label.values()),
                                  configuration={'module_name': 'models.dslim_bert_base_ner',
                                                 'id_to_label_map': id2label,
                                                 'label_to_id_map': {v:k for k,v in id2label.items()}},
                                  project_id=package.project.id
                                  )
    return model


def script():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = "My name is Wolfgang and I live in Berlin"

    ner_results = nlp(example)
    print(ner_results)

    collection = dl.AnnotationCollection()
    for res in ner_results:
        collection.add(dl.Text(text_type='block',
                               label=res['entity'],
                               start=res['start'],
                               end=res['end']),
                       model_info={'name': 'dslim/bert-base-NER',
                                   'confidence': res['score']})
    item = dl.items.get(item_id='635625d0b9cd072b8ccea910')
    item.annotations.upload(collection)
