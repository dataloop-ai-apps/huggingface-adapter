import dtlpy as dl
import logging
import os

logger = logging.getLogger('huggingface-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for HuggingFace models',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """

        module_name = self.model_entity.configuration.get('module_name', 'models.ocr')
        module = __import__(module_name, fromlist=['HuggingAdapter'])
        self.hugging = getattr(module, 'HuggingAdapter')(self.configuration)
        logger.info("Loaded module {!r} successfully".format(module_name))

    def save(self, local_path: str, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'model_filename': 'weights/latest.pt'})

    def prepare_item_func(self, item: dl.Item):
        return self.hugging.prepare_item_func(item)

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        return self.hugging.predict(model_entity=self.model_entity, batch=batch)
