from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from concurrent.futures import ThreadPoolExecutor
import dtlpy as dl
import logging
import torch
import tqdm
import json
import os

logger = logging.getLogger('[NEMO-EMBEDDINGS]')


class Embedder(dl.BaseServiceRunner):

    def __init__(self):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceBgeEmbeddings(cache_folder='.cache',
                                                   model_name=model_name,
                                                   model_kwargs=model_kwargs,
                                                   encode_kwargs=encode_kwargs
                                                   )
        self.name = 'rag_embeddings_BAAI_bge-large-en'

    def create_feature_set(self, project: dl.Project):
        try:
            feature_set = project.feature_sets.get(feature_set_name=self.name)
            logger.info(f'Feature Set found! name: {feature_set.name}, id: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found. creating...')
            feature_set = project.feature_sets.create(name=self.name,
                                                      size=1024,
                                                      set_type='embeddings',
                                                      entity_type=dl.FeatureEntityType.ITEM)
            logger.info(f'Feature Set created! name: {feature_set.name}, id: {feature_set.id}')
        return feature_set

    def embed(self, text: str):
        return text, json.dumps(self.embeddings.embed_documents([text])[0])

    def embed_item(self, item: dl.Item):
        if item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') == 'prompt':
            _json = json.load(item.download(save_locally=False))
            # check if hyde
            is_hyde = _json['metadata'].get('isHyde', None)
            if is_hyde is None:
                is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
            prompt_key = list(_json['prompts'].keys())[0]
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

    @staticmethod
    def upload_single(pbar, feature_set, item, project_id, value):
        try:
            feature_set.features.create(value=value,
                                        project_id=project_id,
                                        entity=item)
        except dl.exceptions.BadRequest:
            # already created
            pass
        except Exception as e:
            print('FAILED', e)
        finally:
            pbar.update()

    def embed_dataset(self, dataset: dl.Dataset, query: dict = None):
        if query is None:
            filters = None
        else:
            filters = dl.Filters(custom_filter=query)
        feature_set = self.create_feature_set(project=dataset.project)
        items = list(dataset.items.list(filters=filters).all())
        pbar = tqdm.tqdm(total=len(items))
        pool = ThreadPoolExecutor(max_workers=32)
        for item in items:
            local_filepath = os.path.join('data', item.filename[1:])
            if not os.path.isfile(local_filepath):
                item.download(local_filepath)
            with open(local_filepath, encoding='utf-8') as f:
                content = f.read()
            content, embeddings = self.embed(content)
            pool.submit(self.upload_single,
                        item=item,
                        feature_set=feature_set,
                        project_id=dataset.project.id,
                        value=json.loads(embeddings),
                        pbar=pbar)
        pool.shutdown()


if __name__ == "__main__":
    dl.setenv('prod')
    runner = Embedder()
    runner.embed_dataset(dataset=dl.datasets.get(dataset_id='66b21da2733517994470c5a0'))
    # runner.embed_item(item=dl.items.get(None, '66564471703b07334efe79d1'))
