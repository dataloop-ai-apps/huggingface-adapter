import dtlpy as dl
import logging
import os
from transformers import Trainer

logger = logging.getLogger('huggingface-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for HuggingFace models',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """

    """
    tokenizer = None

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

    def is_hugging_adapter_training_enabled(self):
        requirements = ['create_datasets', 'create_training_args']
        optionals = ['create_data_collator', 'model_initialization', 'select_tokenizer',
                     'create_callbacks', 'create_optimizers', 'compute_metrics',
                     'preprocess_logits']
        not_implemented = [req for req in requirements if not hasattr(self.hugging, req)]
        if not_implemented:
            raise NotImplementedError(f'Hugging Face adapter missing methods {", ".join(not_implemented)}. Cannot train'
                                      f' without these methods.')
        else:
            training_options = {opt: getattr(self.hugging, opt)() for opt in optionals if hasattr(self.hugging, opt)}
            return training_options

    def train(self, data_path, output_path, **kwargs):
        training_options = self.is_hugging_adapter_training_enabled()
        if len(training_options) >= 0:
            training_mode = self.configuration.get("training_mode", "finetune")
            if training_mode == "finetune" or training_mode == "train":
                train_dataset, eval_dataset = self.hugging.create_datasets(data_path)
                trainer = Trainer(
                        model=self.hugging.model,
                        args=self.hugging.create_training_args(data_path, output_path, **kwargs),
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=training_options.get("compute_metrics"),
                        data_collator=training_options.get("create_data_collator"),
                        model_init=training_options.get("model_initialization"),
                        tokenizer=training_options.get("select_tokenizer"),
                        callbacks=training_options.get("create_callbacks"),
                        optimizers=training_options.get("create_optimizers"),
                        preprocess_logits_for_metrics=training_options.get("preprocess_logits")
                    )
                trainer.train()
                eval_results = trainer.evaluate()
                print("*" * 50)
                print(f"Training results: {eval_results}")
                print("*" * 50)
            elif training_mode == "rlhf":
                raise NotImplementedError("RHLF training still not implemented.")
            else:
                raise Exception("Unrecognized training method")

    def convert_from_dtlpy(self, data_path, **kwargs):
        if hasattr(self.hugging, "convert_from_dtlpy"):
            self.hugging.convert_from_dtlpy(data_path, **kwargs)
