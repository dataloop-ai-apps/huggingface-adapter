import dtlpy as dl
import logging
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger('[HEBERT-EMBEDDER]')


class Embedder(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configuration.get("model_name", "avichr/heBERT"))
        self.model = AutoModel.from_pretrained(self.configuration.get("model_name", "avichr/heBERT"))

    def _embed(self, texts):
        output = list()

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get the model output
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the [CLS] token embedding as the sentence embedding
            output.append(outputs.last_hidden_state[:, 0, :].numpy()[0].tolist())
        return output

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][0]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][0]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            embedding = self._embed(texts=[text])[0]
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings
