import dtlpy as dl
import json
import os
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger('[HEBERT-EMBEDDER]')


class Embedder(dl.BaseServiceRunner):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
        self.model = AutoModel.from_pretrained("avichr/heBERT")
        self.name = 'avichr_heBERT'
        self.feature_set = None

    def create_feature_set(self, project: dl.Project):
        try:
            feature_set = project.feature_sets.get(feature_set_name=self.name)
            logger.info(f'Feature Set found! name: {feature_set.name}, id: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found. creating...')
            feature_set = project.feature_sets.create(name=self.name,
                                                      entity_type=dl.FeatureEntityType.ITEM,
                                                      project_id=project.id,
                                                      set_type='embeddings',
                                                      size=768)
            logger.info(f'Feature Set created! name: {feature_set.name}, id: {feature_set.id}')
        return feature_set

    def __call__(self, texts):
        output = list()

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get the model output
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the [CLS] token embedding as the sentence embedding
            output.append(outputs.last_hidden_state[:, 0, :].numpy()[0].tolist())
        return output

    def embed(self, text):
        output = self(texts=[text])[0]
        return text, output

    def embed_item(self, item: dl.Item):
        if item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') == 'prompt':
            _json = json.load(item.download(save_locally=False))
            # check if hyde
            is_hyde = _json['metadata'].get('isHyde', False)
            prompt_key = list(_json['prompts'].keys())[-1]
            if is_hyde:
                prompt_anns = [a for a in item.annotations.list() if a.metadata['system']['promptId'] == prompt_key]
                hyde_anns = [a for a in prompt_anns if 'hyde' in a.metadata['system']['model']['name'].lower()]
                if len(hyde_anns) == 0:
                    logger.warning('Not hyde found, defaulting to regular')
                    text = [a["value"] for a in _json['prompts'][prompt_key] if 'text' in a['mimetype']][0]
                else:
                    text = hyde_anns[0].coordinates
            else:
                text = [a["value"] for a in _json['prompts'][prompt_key] if 'text' in a['mimetype']][0]
        else:
            raise ValueError('Currently only prompt item is supported ')
        _, embeddings_str = self.embed(text)
        return item, embeddings_str

    def extract_item(self, item: dl.Item):
        path = None
        try:
            path = item.download(overwrite=True)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            output = self(texts=[text])[0]
            if output is not None:
                self.feature_set.features.create(value=output, entity=item)
        except Exception as e:
            logger.exception(e)
        finally:
            if path is not None and os.path.exists(path):
                os.remove(path)
        return item

    def extract_dataset(self, dataset: dl.Dataset, query=None, progress=None):
        pages = dataset.items.list()
        for item in pages.all():
            self.extract_item(item=item)
        return dataset

    def extract_dataset_threaded(self, dataset: dl.Dataset, query=None, progress=None):
        pages = dataset.items.list(filters=query)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.extract_item, obj) for obj in pages.all()]
            done_count = 0
            previous_update = 0
            while futures:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                done_count += len(done)
                current_progress = done_count * 100 // pages.items_count
                if (current_progress // 10) % 10 > previous_update:
                    previous_update = (current_progress // 10) % 10
                    if progress is not None:
                        progress.update(progress=previous_update * 10)
                    else:
                        logger.info(f'Extracted {done_count} out of {pages.items_count} items')


if __name__ == "__main__":
    dl.setenv('prod')
    project = dl.projects.get('Merkava Demo')
    dataset = project.datasets.get(dataset_name='TAKAM')
    self = Embedder()
    # self.extract_dataset_threaded(dataset=dataset)