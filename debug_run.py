from model_adapter import ModelAdapter
import dtlpy as dl


dl.setenv('rc')
# model_entity = dl.models.get(model_id='635559ec622e6d84a3ebd23f')  # facebook/detr-resnet-50-panoptic
model_entity = dl.models.get(model_id='635627dbde3a5b3bf98d013c')  # dslim/bert-base-NER
model_entity.input_type = 'txt'

adapter = ModelAdapter(model_entity=model_entity)

item = dl.items.get(item_id='635625d0b9cd072b8ccea910')
outs = adapter.predict_items([item], with_upload=True)
