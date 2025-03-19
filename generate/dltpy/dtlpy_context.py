import dataclasses
import tempfile
import datetime
import logging
import requests
import string
import shutil
import random
import base64
import tqdm
import json
import enum
import attr
import sys
import io
import os


from PIL import Image
from functools import partial
import numpy as np
import dtlpy as dl
from concurrent.futures import ThreadPoolExecutor

from dtlpy import entities, utilities, repositories, exceptions
from dtlpy.services import service_defaults
from dtlpy.services.api_client import client as client_api


logger = logging.getLogger('dtlpy_context')



class PromptType(str, enum.Enum):
    TEXT = 'application/text'
    IMAGE = 'image/*'
    AUDIO = 'audio/*'
    VIDEO = 'video/*'
    METADATA = 'metadata'


class Prompt:
    def __init__(self, key, role='user'):
        """
        Create a single Prompt. Prompt can contain multiple mimetype elements, e.g. text sentence and an image.
        :param key: unique identifier of the prompt in the item
        """
        self.key = key
        self.elements = list()
        # to avoid broken stream of json files - DAT-75653
        client_api.default_headers['x-dl-sanitize'] = '0'
        self._items = repositories.Items(client_api=client_api)
        self.metadata = {'role': role}

    def add_element(self, value, mimetype='application/text'):
        """

        :param value: url or string of the input
        :param mimetype: mimetype of the input. options: `text`, `image/*`, `video/*`, `audio/*`
        :return:
        """
        allowed_prompt_types = [prompt_type for prompt_type in PromptType]
        if mimetype not in allowed_prompt_types:
            raise ValueError(f'Invalid mimetype: {mimetype}. Allowed values: {allowed_prompt_types}')
        if mimetype == PromptType.METADATA and isinstance(value, dict):
            self.metadata.update(value)
        else:
            self.elements.append({'mimetype': mimetype,
                                  'value': value})

    def to_json(self):
        """
        Convert Prompt entity to the item json

        :return:
        """
        elements_json = [
            {
                "mimetype": e['mimetype'],
                "value": e['value'],
            } for e in self.elements if not e['mimetype'] == PromptType.METADATA
        ]
        elements_json.append({
            "mimetype": PromptType.METADATA,
            "value": self.metadata
        })
        return {
            self.key: elements_json
        }

    def _convert_stream_to_binary(self, image_url: str):
        """
        Convert a stream to binary
        :param image_url: dataloop image stream url
        :return: binary object
        """
        image_buffer = None
        if '.' in image_url and 'dataloop.ai' not in image_url:
            # URL and not DL item stream
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Check for valid image content type
                if response.headers["Content-Type"].startswith("image/"):
                    # Read the image data in chunks to avoid loading large images in memory
                    image_buffer = b"".join(chunk for chunk in response.iter_content(1024))
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download image from URL: {image_url}, error: {e}")

        elif '.' in image_url and 'stream' in image_url:
            # DL Stream URL
            item_id = image_url.split("/stream")[0].split("/items/")[-1]
            image_buffer = self._items.get(item_id=item_id).download(save_locally=False).getvalue()
        else:
            # DL item ID
            image_buffer = self._items.get(item_id=image_url).download(save_locally=False).getvalue()

        if image_buffer is not None:
            encoded_image = base64.b64encode(image_buffer).decode()
        else:
            logger.error(f'Invalid image url: {image_url}')
            return None

        return f'data:image/jpeg;base64,{encoded_image}'

    def messages(self):
        """
        return a list of messages in the prompt item,
        messages are returned following the openai SDK format https://platform.openai.com/docs/guides/vision
        """
        messages = []
        for element in self.elements:
            if element['mimetype'] == PromptType.TEXT:
                data = {
                    "type": "text",
                    "text": element['value']
                }
                messages.append(data)
            elif element['mimetype'] == PromptType.IMAGE:
                image_url = self._convert_stream_to_binary(element['value'])
                data = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
                messages.append(data)
            elif element['mimetype'] == PromptType.AUDIO:
                raise NotImplementedError('Audio prompt is not supported yet')
            elif element['mimetype'] == PromptType.VIDEO:
                raise NotImplementedError('Video prompt is not supported yet')
            else:
                raise ValueError(f'Invalid mimetype: {element["mimetype"]}')
        return messages, self.key


class PromptItem:
    def __init__(self, name, item: entities.Item = None, role_mapping=None):
        if role_mapping is None:
            role_mapping = {'user': 'item',
                            'assistant': 'annotation'}
        if not isinstance(role_mapping, dict):
            raise ValueError(f'input role_mapping must be dict. type: {type(role_mapping)}')
        self.role_mapping = role_mapping
        # prompt item name
        self.name = name
        # list of user prompts in the prompt item
        self.prompts = list()
        self.assistant_prompts = list()
        # list of assistant (annotations) prompts in the prompt item
        # Dataloop Item
        self._messages = []
        self._item: entities.Item = item
        self._annotations: entities.AnnotationCollection = None
        if item is not None:
            self._items = item.items
            self.fetch()
        else:
            self._items = repositories.Items(client_api=client_api)
        # to avoid broken stream of json files - DAT-75653
        self._items._client_api.default_headers['x-dl-sanitize'] = '0'

    @classmethod
    def from_messages(cls, messages: list):
        ...

    @classmethod
    def from_item(cls, item: entities.Item):
        """
        Load a prompt item from the platform
        :param item : Item object
        :return: PromptItem object
        """
        if 'json' not in item.mimetype or item.system.get('shebang', dict()).get('dltype') != 'prompt':
            raise ValueError('Expecting a json item with system.shebang.dltype = prompt')
        return cls(name=item.name, item=item)

    @classmethod
    def from_local_file(cls, filepath):
        """
        Create a new prompt item from a file
        :param filepath: path to the file
        :return: PromptItem object
        """
        if os.path.exists(filepath) is False:
            raise FileNotFoundError(f'File does not exists: {filepath}')
        if 'json' not in os.path.splitext(filepath)[-1]:
            raise ValueError(f'Expected path to json item, got {os.path.splitext(filepath)[-1]}')
        prompt_item = cls(name=filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt_item.prompts = prompt_item._load_item_prompts(data=data)
        return prompt_item

    @staticmethod
    def _load_item_prompts(data):
        prompts = list()
        for prompt_key, prompt_elements in data.get('prompts', dict()).items():
            content = list()
            for element in prompt_elements:
                content.append({'value': element.get('value', dict()),
                                'mimetype': element['mimetype']})
            prompt = Prompt(key=prompt_key, role="user")
            for element in content:
                prompt.add_element(value=element.get('value', ''),
                                   mimetype=element.get('mimetype', PromptType.TEXT))
            prompts.append(prompt)
        return prompts

    @staticmethod
    def _load_annotations_prompts(annotations: entities.AnnotationCollection):
        """
        Get all the annotations in the item for the assistant messages
        """
        # clearing the assistant prompts from previous annotations that might not belong
        assistant_prompts = list()
        for annotation in annotations:
            prompt_id = annotation.metadata.get('system', dict()).get('promptId', None)
            model_info = annotation.metadata.get('user', dict()).get('model', dict())
            annotation_id = annotation.id
            if annotation.type == 'ref_image':
                prompt = Prompt(key=prompt_id, role='assistant')
                prompt.add_element(value=annotation.annotation_definition.coordinates.get('ref'),
                                   mimetype=PromptType.IMAGE)
            elif annotation.type == 'text':
                prompt = Prompt(key=prompt_id, role='assistant')
                prompt.add_element(value=annotation.annotation_definition.coordinates,
                                   mimetype=PromptType.TEXT)
            else:
                raise ValueError(f"Unsupported annotation type: {annotation.type}")

            prompt.add_element(value={'id': annotation_id,
                                      'model_info': model_info},
                               mimetype=PromptType.METADATA)
            assistant_prompts.append(prompt)
        return assistant_prompts

    def to_json(self):
        """
        Convert the entity to a platform item.

        :return:
        """
        prompts_json = {
            "shebang": "dataloop",
            "metadata": {
                "dltype": 'prompt'
            },
            "prompts": {}
        }
        for prompt in self.prompts:
            for prompt_key, prompt_values in prompt.to_json().items():
                prompts_json["prompts"][prompt_key] = prompt_values
        return prompts_json

    def to_messages(self, model_name=None, include_assistant=True):
        all_prompts_messages = dict()
        for prompt in self.prompts:
            if prompt.key not in all_prompts_messages:
                all_prompts_messages[prompt.key] = list()
            prompt_messages, prompt_key = prompt.messages()
            messages = {
                'role': prompt.metadata.get('role', 'user'),
                'content': prompt_messages
            }
            all_prompts_messages[prompt.key].append(messages)
        if include_assistant is True:
            # reload to filer model annotations
            for prompt in self.assistant_prompts:
                prompt_model_name = prompt.metadata.get('model_info', dict()).get('name')
                if model_name is not None and prompt_model_name != model_name:
                    continue
                if prompt.key not in all_prompts_messages:
                    logger.warning(
                        f'Prompt key {prompt.key} is not found in the user prompts, skipping Assistant prompt')
                    continue
                prompt_messages, prompt_key = prompt.messages()
                assistant_messages = {
                    'role': 'assistant',
                    'content': prompt_messages
                }
                all_prompts_messages[prompt.key].append(assistant_messages)
        res = list()
        for prompts in all_prompts_messages.values():
            for prompt in prompts:
                res.append(prompt)
        self._messages = res
        return self._messages

    def to_bytes_io(self):
        # Used for item upload, do not delete
        byte_io = io.BytesIO()
        byte_io.name = self.name
        byte_io.write(json.dumps(self.to_json()).encode())
        byte_io.seek(0)
        return byte_io

    def fetch(self):
        if self._item is None:
            raise ValueError('Missing item, nothing to fetch..')
        self._item = self._items.get(item_id=self._item.id)
        self._annotations = self._item.annotations.list()
        self.prompts = self._load_item_prompts(data=json.load(self._item.download(save_locally=False)))
        self.assistant_prompts = self._load_annotations_prompts(self._annotations)

    def build_context(self, nearest_items, add_metadata=None) -> str:
        """
        Create a context stream from nearest items list.
        add_metadata is a list of location in the item.metadata to add to the context, for instance ['system.document.source']
        :param nearest_items: list of item ids
        :param add_metadata: list of metadata location to add metadata to context
        :return:
        """
        if add_metadata is None:
            add_metadata = list()

        def stream_single(w_id):
            context_item = self._items.get(item_id=w_id)
            buf = context_item.download(save_locally=False)
            text = buf.read().decode(encoding='utf-8')
            m = ""
            for path in add_metadata:
                parts = path.split('.')
                value = context_item.metadata
                part = ""
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = ""

                m += f"{part}:{value}\n"
            return text, m

        pool = ThreadPoolExecutor(max_workers=32)
        context = ""
        if len(nearest_items) > 0:
            # build context
            results = pool.map(stream_single, nearest_items)
            for res in results:
                context += f"\n<source>\n{res[1]}\n</source>\n<text>\n{res[0]}\n</text>"
        return context

    def add(self,
            message: dict,
            prompt_key: str = None,
            model_info: dict = None):
        """
        add a prompt to the prompt item
        prompt: a dictionary. keys are prompt message id, values are prompt messages
        responses: a list of annotations representing responses to the prompt

        :param message:
        :param prompt_key:
        :param model_info:
        :return:
        """
        role = message.get('role', 'user')
        content = message.get('content', list())

        if self.role_mapping.get(role, 'item') == 'item':
            if prompt_key is None:
                prompt_key = str(len(self.prompts) + 1)
            # for new prompt we need a new key
            prompt = Prompt(key=prompt_key, role=role)
            for element in content:
                prompt.add_element(value=element.get('value', ''),
                                   mimetype=element.get('mimetype', PromptType.TEXT))

            # create new prompt and add to prompts
            self.prompts.append(prompt)
            if self._item is not None:
                self._item._Item__update_item_binary(_json=self.to_json())
        else:
            if prompt_key is None:
                prompt_key = str(len(self.prompts))
            assistant_message = content[0]
            assistant_mimetype = assistant_message.get('mimetype', PromptType.TEXT)
            uploaded_annotation = None

            # find if prompt
            if model_info is None:
                # dont search for existing if there's no model information
                existing_prompt = None
            else:
                existing_prompts = list()
                for prompt in self.assistant_prompts:
                    prompt_id = prompt.key
                    model_name = prompt.metadata.get('model_info', dict()).get('name')
                    if prompt_id == prompt_key and model_name == model_info.get('name'):
                        # TODO how to handle multiple annotations
                        existing_prompts.append(prompt)
                if len(existing_prompts) > 1:
                    assert False, "shouldn't be here! more than 1 annotation for a single model"
                elif len(existing_prompts) == 1:
                    # found model annotation to upload
                    existing_prompt = existing_prompts[0]
                else:
                    # no annotation found
                    existing_prompt = None

            if existing_prompt is None:
                prompt = Prompt(key=prompt_key)
                if assistant_mimetype == PromptType.TEXT:
                    annotation_definition = entities.FreeText(text=assistant_message.get('value'))
                    prompt.add_element(value=annotation_definition.to_coordinates(None),
                                       mimetype=PromptType.TEXT)
                elif assistant_mimetype == PromptType.IMAGE:
                    annotation_definition = entities.RefImage(ref=assistant_message.get('value'))
                    prompt.add_element(value=annotation_definition.to_coordinates(None).get('ref'),
                                       mimetype=PromptType.IMAGE)
                else:
                    raise NotImplementedError('Only images of mimetype image and text are supported')
                metadata = {'system': {'promptId': prompt_key},
                            'user': {'model': model_info}}
                prompt.add_element(mimetype=PromptType.METADATA,
                                   value={"model_info": model_info})

                existing_annotation = entities.Annotation.new(item=self._item,
                                                              metadata=metadata,
                                                              annotation_definition=annotation_definition)
                uploaded_annotation = existing_annotation.upload()
                prompt.add_element(mimetype=PromptType.METADATA,
                                   value={"id": uploaded_annotation.id})
                existing_prompt = prompt
                self.assistant_prompts.append(prompt)

            existing_prompt_element = [element for element in existing_prompt.elements if
                                       element['mimetype'] != PromptType.METADATA][-1]
            existing_prompt_element['value'] = assistant_message.get('value')
            if uploaded_annotation is None:
                # Creating annotation with old dict to match platform dict
                annotation_definition = entities.FreeText(text='')
                metadata = {'system': {'promptId': prompt_key},
                            'user': {'model': existing_prompt.metadata.get('model_info')}}
                annotation = entities.Annotation.new(item=self._item,
                                                     metadata=metadata,
                                                     annotation_definition=annotation_definition
                                                     )
                annotation.id = existing_prompt.metadata['id']
                # set the platform dict to match the old annotation for the dict difference check, otherwise it won't
                # update
                annotation._platform_dict = annotation.to_json()
                # update the annotation with the new text
                annotation.annotation_definition.text = existing_prompt_element['value']
                self._item.annotations.update(annotation)

    def update(self):
        """
        Update the prompt item in the platform. 
        """
        if self._item is not None:
            self._item._Item__update_item_binary(_json=self.to_json())
        else:
            raise ValueError('Cannot update PromptItem without an item.')


@dataclasses.dataclass
class AdapterDefaults(dict):
    # for predict items, dataset, evaluate
    upload_annotations: bool = dataclasses.field(default=True)
    clean_annotations: bool = dataclasses.field(default=True)
    # for embeddings
    upload_features: bool = dataclasses.field(default=True)
    # for training
    root_path: str = dataclasses.field(default=None)
    data_path: str = dataclasses.field(default=None)
    output_path: str = dataclasses.field(default=None)

    def __post_init__(self):
        # Initialize the internal dictionary with the dataclass fields
        self.update(**dataclasses.asdict(self))

    def update(self, **kwargs):
        for f in dataclasses.fields(AdapterDefaults):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])
        super().update(**kwargs)

    def resolve(self, key, *args):

        for arg in args:
            if arg is not None:
                return arg
        return self.get(key, None)


class BaseModelAdapter(utilities.BaseServiceRunner):
    _client_api = attr.ib(type=client_api, repr=False)

    def __init__(self, model_entity: entities.Model = None):
        self.adapter_defaults = AdapterDefaults()
        self.logger = logger
        # entities
        self._model_entity = None
        self._package = None
        self._base_configuration = dict()
        self.package_name = None
        self.model = None
        self.bucket_path = None
        # funcs
        self.item_to_batch_mapping = {'text': self._item_to_text,
                                      'image': self._item_to_image}
        if model_entity is not None:
            self.load_from_model(model_entity=model_entity)
        logger.warning(
            "in case of a mismatch between 'model.name' and 'model_info.name' in the model adapter, model_info.name will be updated to align with 'model.name'.")

    ##################
    # Configurations #
    ##################

    @property
    def configuration(self) -> dict:
        # load from model
        if self._model_entity is not None:
            configuration = self.model_entity.configuration
        # else - load the default from the package
        elif self._package is not None:
            configuration = self.package.metadata.get('system', {}).get('ml', {}).get('defaultConfiguration', {})
        else:
            configuration = self._base_configuration
        return configuration

    @configuration.setter
    def configuration(self, d):
        assert isinstance(d, dict)
        if self._model_entity is not None:
            self._model_entity.configuration = d

    ############
    # Entities #
    ############
    @property
    def model_entity(self):
        if self._model_entity is None:
            raise ValueError(
                "No model entity loaded. Please load a model (adapter.load_from_model(<dl.Model>)) or set: 'adapter.model_entity=<dl.Model>'")
        assert isinstance(self._model_entity, entities.Model)
        return self._model_entity

    @model_entity.setter
    def model_entity(self, model_entity):
        assert isinstance(model_entity, entities.Model)
        if self._model_entity is not None and isinstance(self._model_entity, entities.Model):
            if self._model_entity.id != model_entity.id:
                self.logger.warning(
                    'Replacing Model from {!r} to {!r}'.format(self._model_entity.name, model_entity.name))
        self._model_entity = model_entity
        self.package = model_entity.package

    @property
    def package(self):
        if self._model_entity is not None:
            self.package = self.model_entity.package
        if self._package is None:
            raise ValueError('Missing Package entity on adapter. Please set: "adapter.package=package"')
        assert isinstance(self._package, (entities.Package, entities.Dpk))
        return self._package

    @package.setter
    def package(self, package):
        assert isinstance(package, (entities.Package, entities.Dpk))
        self.package_name = package.name
        self._package = package

    ###################################
    # NEED TO IMPLEMENT THESE METHODS #
    ###################################

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        raise NotImplementedError("Please implement `load` method in {}".format(self.__class__.__name__))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        raise NotImplementedError("Please implement `save` method in {}".format(self.__class__.__name__))

    def train(self, data_path, output_path, **kwargs):
        """
        Virtual method - need to implement

        Train the model according to data in data_paths and save the train outputs to output_path,
        this include the weights and any other artifacts created during train

        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        raise NotImplementedError("Please implement `train` method in {}".format(self.__class__.__name__))

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of items

            Virtual method - need to implement

        :param batch: output of the `prepare_item_func` func
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        raise NotImplementedError("Please implement `predict` method in {}".format(self.__class__.__name__))

    def embed(self, batch, **kwargs):
        """ Extract model embeddings on batch of items

            Virtual method - need to implement

        :param batch: output of the `prepare_item_func` func
        :return: `list[list]` a feature vector per each item in the batch
        """
        raise NotImplementedError("Please implement `embed` method in {}".format(self.__class__.__name__))

    def evaluate(self, model: entities.Model, dataset: entities.Dataset, filters: entities.Filters) -> entities.Model:
        """
        This function evaluates the model prediction on a dataset (with GT annotations).
        The evaluation process will upload the scores and metrics to the platform.

        :param model: The model to evaluate (annotation.metadata.system.model.name
        :param dataset: Dataset where the model predicted and uploaded its annotations
        :param filters: Filters to query items on the dataset
        :return:
        """
        import dtlpymetrics
        compare_types = model.output_type
        if not filters:
            filters = entities.Filters()
        if filters is not None and isinstance(filters, dict):
            filters = entities.Filters(custom_filter=filters)
        model = dtlpymetrics.scoring.create_model_score(model=model,
                                                        dataset=dataset,
                                                        filters=filters,
                                                        compare_types=compare_types)
        return model

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """
        raise NotImplementedError("Please implement `convert_from_dtlpy` method in {}".format(self.__class__.__name__))

    #################
    # DTLPY METHODS #
    ################
    def prepare_item_func(self, item: entities.Item):
        """
        Prepare the Dataloop item before calling the `predict` function with a batch.
        A user can override this function to load item differently
        Default will load the item according the input_type (mapping type to function is in self.item_to_batch_mapping)

        :param item:
        :return: preprocessed: the var with the loaded item information (e.g. ndarray for image, dict for json files etc)
        """
        # Item to batch func
        if isinstance(self.model_entity.input_type, list):
            if 'text' in self.model_entity.input_type and 'text' in item.mimetype:
                processed = self._item_to_text(item)
            elif 'image' in self.model_entity.input_type and 'image' in item.mimetype:
                processed = self._item_to_image(item)
            else:
                processed = self._item_to_item(item)

        elif self.model_entity.input_type in self.item_to_batch_mapping:
            processed = self.item_to_batch_mapping[self.model_entity.input_type](item)

        else:
            processed = self._item_to_item(item)

        return processed

    def prepare_data(self,
                     dataset: entities.Dataset,
                     # paths
                     root_path=None,
                     data_path=None,
                     output_path=None,
                     #
                     overwrite=False,
                     **kwargs):
        """
        Prepares dataset locally before training or evaluation.
        download the specific subset selected to data_path and preforms `self.convert` to the data_path dir

        :param dataset: dl.Dataset
        :param root_path: `str` root directory for training. default is "tmp". Can be set using self.adapter_defaults.root_path
        :param data_path: `str` dataset directory. default <root_path>/"data". Can be set using self.adapter_defaults.data_path
        :param output_path: `str` save everything to this folder. default <root_path>/"output". Can be set using self.adapter_defaults.output_path

        :param bool overwrite: overwrite the data path (download again). default is False
        """
        # define paths
        dataloop_path = service_defaults.DATALOOP_PATH
        root_path = self.adapter_defaults.resolve("root_path", root_path)
        data_path = self.adapter_defaults.resolve("data_path", data_path)
        output_path = self.adapter_defaults.resolve("output_path", output_path)

        if root_path is None:
            now = datetime.datetime.now()
            root_path = os.path.join(dataloop_path,
                                     'model_data',
                                     "{s_id}_{s_n}".format(s_id=self.model_entity.id, s_n=self.model_entity.name),
                                     now.strftime('%Y-%m-%d-%H%M%S'),
                                     )
        if data_path is None:
            data_path = os.path.join(root_path, 'datasets', self.model_entity.dataset.id)
            os.makedirs(data_path, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(root_path, 'output')
            os.makedirs(output_path, exist_ok=True)

        if len(os.listdir(data_path)) > 0:
            self.logger.warning("Data path directory ({}) is not empty..".format(data_path))

        annotation_options = entities.ViewAnnotationOptions.JSON
        if self.model_entity.output_type in [entities.AnnotationType.SEGMENTATION]:
            annotation_options = entities.ViewAnnotationOptions.INSTANCE

        # Download the subset items
        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if subsets is None:
            raise ValueError("Model (id: {}) must have subsets in metadata.system.subsets".format(self.model_entity.id))
        for subset, filters_dict in subsets.items():
            filters = entities.Filters(custom_filter=filters_dict)
            data_subset_base_path = os.path.join(data_path, subset)
            if os.path.isdir(data_subset_base_path) and not overwrite:
                # existing and dont overwrite
                self.logger.debug("Subset {!r} already exists (and overwrite=False). Skipping.".format(subset))
            else:
                self.logger.debug("Downloading subset {!r} of {}".format(subset,
                                                                         self.model_entity.dataset.name))

                if self.model_entity.output_type is not None:
                    if self.model_entity.output_type in [entities.AnnotationType.SEGMENTATION,
                                                         entities.AnnotationType.POLYGON]:
                        model_output_types = [entities.AnnotationType.SEGMENTATION, entities.AnnotationType.POLYGON]
                    else:
                        model_output_types = [self.model_entity.output_type]
                    annotation_filters = entities.Filters(
                        field=entities.FiltersKnownFields.TYPE,
                        values=model_output_types,
                        resource=entities.FiltersResource.ANNOTATION,
                        operator=entities.FiltersOperations.IN
                    )
                else:
                    annotation_filters = entities.Filters(resource=entities.FiltersResource.ANNOTATION)

                if not self.configuration.get("include_model_annotations", False):
                    annotation_filters.add(
                        field="metadata.system.model.name",
                        values=False,
                        operator=entities.FiltersOperations.EXISTS
                    )

                ret_list = dataset.items.download(filters=filters,
                                                  local_path=data_subset_base_path,
                                                  annotation_options=annotation_options,
                                                  annotation_filters=annotation_filters
                                                  )

        self.convert_from_dtlpy(data_path=data_path, **kwargs)
        return root_path, data_path, output_path

    def load_from_model(self, model_entity=None, local_path=None, overwrite=True, **kwargs):
        """ Loads a model from given `dl.Model`.
            Reads configurations and instantiate self.model_entity
            Downloads the model_entity bucket (if available)

        :param model_entity:  `str` dl.Model entity
        :param local_path:  `str` directory path in local FileSystem to download the model_entity to
        :param overwrite: `bool` (default False) if False does not download files with same name else (True) download all
        """
        if model_entity is not None:
            self.model_entity = model_entity
        if local_path is None:
            local_path = os.path.join(service_defaults.DATALOOP_PATH, "models", self.model_entity.name)
        # Load configuration
        self.configuration = self.model_entity.configuration
        # Update the adapter config with the model config to run over defaults if needed
        self.adapter_defaults.update(**self.configuration)
        # Download
        self.model_entity.artifacts.download(
            local_path=local_path,
            overwrite=overwrite
        )
        self.load(local_path, **kwargs)

    def save_to_model(self, local_path=None, cleanup=False, replace=True, **kwargs):
        """
        Saves the model state to a new bucket and configuration

        Saves configuration and weights to artifacts
        Mark the model as `trained`
        loads only applies for remote buckets

        :param local_path: `str` directory path in local FileSystem to save the current model bucket (weights) (default will create a temp dir)
        :param replace: `bool` will clean the bucket's content before uploading new files
        :param cleanup: `bool` if True (default) remove the data from local FileSystem after upload
        :return:
        """

        if local_path is None:
            local_path = tempfile.mkdtemp(prefix="model_{}".format(self.model_entity.name))
            self.logger.debug("Using temporary dir at {}".format(local_path))

        self.save(local_path=local_path, **kwargs)

        if self.model_entity is None:
            raise ValueError('Missing model entity on the adapter. '
                             'Please set before saving: "adapter.model_entity=model"')

        self.model_entity.artifacts.upload(filepath=os.path.join(local_path, '*'),
                                           overwrite=True)
        if cleanup:
            shutil.rmtree(path=local_path, ignore_errors=True)
            self.logger.info("Clean-up. deleting {}".format(local_path))

    # ===============
    # SERVICE METHODS
    # ===============

    @entities.Package.decorators.function(display_name='Predict Items',
                                          inputs={'items': 'Item[]'},
                                          outputs={'items': 'Item[]', 'annotations': 'Annotation[]'})
    def predict_items(self, items: list, upload_annotations=None, clean_annotations=None, batch_size=None, **kwargs):
        """
        Run the predict function on the input list of items (or single) and return the items and the predictions.
        Each prediction is by the model output type (package.output_type) and model_info in the metadata

        :param items: `List[dl.Item]` list of items to predict
        :param upload_annotations: `bool` uploads the predictions on the given items
        :param clean_annotations: `bool` deletes previous model annotations (predictions) before uploading new ones
        :param batch_size: `int` size of batch to run a single inference

        :return: `List[dl.Item]`, `List[List[dl.Annotation]]`
        """
        if batch_size is None:
            batch_size = self.configuration.get('batch_size', 4)
        upload_annotations = self.adapter_defaults.resolve("upload_annotations", upload_annotations)
        clean_annotations = self.adapter_defaults.resolve("clean_annotations", clean_annotations)
        input_type = self.model_entity.input_type
        self.logger.debug(
            "Predicting {} items, using batch size {}. input type: {}".format(len(items), batch_size, input_type))
        pool = ThreadPoolExecutor(max_workers=16)

        annotations = list()
        for i_batch in tqdm.tqdm(range(0, len(items), batch_size), desc='predicting', unit='bt', leave=None,
                                 file=sys.stdout):
            batch_items = items[i_batch: i_batch + batch_size]
            batch = list(pool.map(self.prepare_item_func, batch_items))
            batch_collections = self.predict(batch, **kwargs)
            _futures = list(pool.map(partial(self._update_predictions_metadata),
                                     batch_items,
                                     batch_collections))
            # Loop over the futures to make sure they are all done to avoid race conditions
            _ = [_f for _f in _futures]
            if upload_annotations is True:
                self.logger.debug(
                    "Uploading items' annotation for model {!r}.".format(self.model_entity.name))
                try:
                    batch_collections = list(pool.map(partial(self._upload_model_annotations,
                                                              clean_annotations=clean_annotations),
                                                      batch_items,
                                                      batch_collections))
                except Exception as err:
                    self.logger.exception("Failed to upload annotations items.")

            for collection in batch_collections:
                # function needs to return `List[List[dl.Annotation]]`
                # convert annotation collection to a list of dl.Annotation for each batch
                if isinstance(collection, entities.AnnotationCollection):
                    annotations.extend([annotation for annotation in collection.annotations])
                else:
                    logger.warning(f'RETURN TYPE MAY BE INVALID: {type(collection)}')
                    annotations.extend(collection)
            # TODO call the callback

        pool.shutdown()
        return items, annotations

    @entities.Package.decorators.function(display_name='Embed Items',
                                          inputs={'items': 'Item[]'},
                                          outputs={'items': 'Item[]', 'features': 'Json[]'})
    def embed_items(self, items: list, upload_features=None, batch_size=None, **kwargs):
        """
        Extract feature from an input list of items (or single) and return the items and the feature vector.

        :param items: `List[dl.Item]` list of items to embed
        :param upload_features: `bool` uploads the features on the given items
        :param batch_size: `int` size of batch to run a single embed

        :return: `List[dl.Item]`, `List[List[vector]]`
        """
        if batch_size is None:
            batch_size = self.configuration.get('batch_size', 4)
        upload_features = self.adapter_defaults.resolve("upload_features", upload_features)
        input_type = self.model_entity.input_type
        self.logger.debug(
            "Embedding {} items, using batch size {}. input type: {}".format(len(items), batch_size, input_type))

        # Search for existing feature set for this model id
        feature_set = self.model_entity.feature_set
        if feature_set is None:
            logger.info('Feature Set not found. creating... ')
            try:
                self.model_entity.project.feature_sets.get(feature_set_name=self.model_entity.name)
                feature_set_name = f"{self.model_entity.name}-{''.join(random.choices(string.ascii_letters + string.digits, k=5))}"
                logger.warning(
                    f"Feature set with the model name already exists. Creating new feature set with name {feature_set_name}")
            except exceptions.NotFound:
                feature_set_name = self.model_entity.name
            feature_set = self.model_entity.project.feature_sets.create(name=feature_set_name,
                                                                        entity_type=entities.FeatureEntityType.ITEM,
                                                                        model_id=self.model_entity.id,
                                                                        project_id=self.model_entity.project_id,
                                                                        set_type=self.model_entity.name,
                                                                        size=self.configuration.get('embeddings_size',
                                                                                                    256))
            logger.info(f'Feature Set created! name: {feature_set.name}, id: {feature_set.id}')
        else:
            logger.info(f'Feature Set found! name: {feature_set.name}, id: {feature_set.id}')

        # upload the feature vectors
        pool = ThreadPoolExecutor(max_workers=16)
        vectors = list()
        for i_batch in tqdm.tqdm(range(0, len(items), batch_size),
                                 desc='embedding',
                                 unit='bt',
                                 leave=None,
                                 file=sys.stdout):
            batch_items = items[i_batch: i_batch + batch_size]
            batch = list(pool.map(self.prepare_item_func, batch_items))
            batch_vectors = self.embed(batch, **kwargs)
            vectors.extend(batch_vectors)
            if upload_features is True:
                self.logger.debug(
                    "Uploading items' feature vectors for model {!r}.".format(self.model_entity.name))
                try:
                    list(pool.map(partial(self._upload_model_features,
                                              feature_set.id,
                                              self.model_entity.project_id),
                                      batch_items,
                                      batch_vectors))
                except Exception as err:
                    self.logger.exception("Failed to upload feature vectors to items.")

        pool.shutdown()
        return items, vectors

    @entities.Package.decorators.function(display_name='Embed Dataset with DQL',
                                          inputs={'dataset': 'Dataset',
                                                  'filters': 'Json'})
    def embed_dataset(self,
                      dataset: entities.Dataset,
                      filters: entities.Filters = None,
                      upload_features=None,
                      batch_size=None,
                      **kwargs):
        """
        Extract feature from all items given

        :param dataset: Dataset entity to predict
        :param filters: Filters entity for a filtering before embedding
        :param upload_features: `bool` uploads the features back to the given items
        :param batch_size: `int` size of batch to run a single embed

        :return: `bool` indicating if the embedding process completed successfully
        """
        if batch_size is None:
            batch_size = self.configuration.get('batch_size', 4)
        upload_features = self.adapter_defaults.resolve("upload_features", upload_features)

        self.logger.debug("Creating embeddings for dataset (name:{}, id:{}), using batch size {}".format(dataset.name,
                                                                                                         dataset.id,
                                                                                                         batch_size))
        if not filters:
            filters = entities.Filters()
        if filters is not None and isinstance(filters, dict):
            filters = entities.Filters(custom_filter=filters)
        pages = dataset.items.list(filters=filters, page_size=batch_size)
        # Item type is 'file' only, can be deleted if default filters are added to custom filters
        items = [item for page in pages for item in page if item.type == 'file']
        self.embed_items(items=items,
                         upload_features=upload_features,
                         batch_size=batch_size,
                         **kwargs)
        return True

    @entities.Package.decorators.function(display_name='Predict Dataset with DQL',
                                          inputs={'dataset': 'Dataset',
                                                  'filters': 'Json'})
    def predict_dataset(self,
                        dataset: entities.Dataset,
                        filters: entities.Filters = None,
                        upload_annotations=None,
                        clean_annotations=None,
                        batch_size=None,
                        **kwargs):
        """
        Predicts all items given

        :param dataset: Dataset entity to predict
        :param filters: Filters entity for a filtering before predicting
        :param upload_annotations: `bool` uploads the predictions back to the given items
        :param clean_annotations: `bool` if set removes existing predictions with the same package-model name (default: False)
        :param batch_size: `int` size of batch to run a single inference

        :return: `bool` indicating if the prediction process completed successfully
        """

        if batch_size is None:
            batch_size = self.configuration.get('batch_size', 4)

        self.logger.debug("Predicting dataset (name:{}, id:{}, using batch size {}".format(dataset.name,
                                                                                           dataset.id,
                                                                                           batch_size))
        if not filters:
            filters = entities.Filters()
        if filters is not None and isinstance(filters, dict):
            filters = entities.Filters(custom_filter=filters)
        pages = dataset.items.list(filters=filters, page_size=batch_size)
        # Item type is 'file' only, can be deleted if default filters are added to custom filters
        items = [item for page in pages for item in page if item.type == 'file']
        self.predict_items(items=items,
                           upload_annotations=upload_annotations,
                           clean_annotations=clean_annotations,
                           batch_size=batch_size,
                           **kwargs)
        return True

    @entities.Package.decorators.function(display_name='Train a Model',
                                          inputs={'model': entities.Model},
                                          outputs={'model': entities.Model})
    def train_model(self,
                    model: entities.Model,
                    cleanup=False,
                    progress: utilities.Progress = None,
                    context: utilities.Context = None):
        """
        Train on existing model.
        data will be taken from dl.Model.datasetId
        configuration is as defined in dl.Model.configuration
        upload the output the model's bucket (model.bucket)
        """
        if isinstance(model, dict):
            model = repositories.Models(client_api=self._client_api).get(model_id=model['id'])
        output_path = None
        try:
            logger.info("Received {s} for training".format(s=model.id))
            model = model.wait_for_model_ready()
            if model.status == 'failed':
                raise ValueError("Model is in failed state, cannot train.")

            ##############
            # Set status #
            ##############
            model.status = 'training'
            if context is not None:
                if 'system' not in model.metadata:
                    model.metadata['system'] = dict()
            model.update()

            ##########################
            # load model and weights #
            ##########################
            logger.info("Loading Adapter with: {n} ({i!r})".format(n=model.name, i=model.id))
            self.load_from_model(model_entity=model)

            ################
            # prepare data #
            ################
            root_path, data_path, output_path = self.prepare_data(
                dataset=self.model_entity.dataset,
                root_path=os.path.join('tmp', model.id)
            )
            # Start the Train
            logger.info("Training {p_name!r} with model {m_name!r} on data {d_path!r}".
                        format(p_name=self.package_name, m_name=model.id, d_path=data_path))
            if progress is not None:
                progress.update(message='starting training')

            def on_epoch_end_callback(i_epoch, n_epoch):
                if progress is not None:
                    progress.update(progress=int(100 * (i_epoch + 1) / n_epoch),
                                    message='finished epoch: {}/{}'.format(i_epoch, n_epoch))

            self.train(data_path=data_path,
                       output_path=output_path,
                       on_epoch_end_callback=on_epoch_end_callback)
            if progress is not None:
                progress.update(message='saving model',
                                progress=99)

            self.save_to_model(local_path=output_path, replace=True)
            model.status = 'trained'
            model.update()
            ###########
            # cleanup #
            ###########
            if cleanup:
                shutil.rmtree(output_path, ignore_errors=True)
        except Exception:
            # save also on fail
            if output_path is not None:
                self.save_to_model(local_path=output_path, replace=True)
            logger.info('Execution failed. Setting model.status to failed')
            model.status = 'failed'
            model.update()
            raise
        return model

    @entities.Package.decorators.function(display_name='Evaluate a Model',
                                          inputs={'model': entities.Model,
                                                  'dataset': entities.Dataset,
                                                  'filters': 'Json'},
                                          outputs={'model': entities.Model,
                                                   'dataset': entities.Dataset
                                                   })
    def evaluate_model(self,
                       model: entities.Model,
                       dataset: entities.Dataset,
                       filters: entities.Filters = None,
                       #
                       progress: utilities.Progress = None,
                       context: utilities.Context = None):
        """
        Evaluate a model.
        data will be downloaded from the dataset and query
        configuration is as defined in dl.Model.configuration
        upload annotations and calculate metrics vs GT

        :param model: Model entity to run prediction
        :param dataset: Dataset to evaluate
        :param filters: Filter for specific items from dataset
        :param progress: dl.Progress for report FaaS progress
        :param context:
        :return:
        """
        logger.info(
            f"Received model: {model.id} for evaluation on dataset (name: {dataset.name}, id: {dataset.id}")
        ##########################
        # load model and weights #
        ##########################
        logger.info(f"Loading Adapter with: {model.name} ({model.id!r})")
        self.load_from_model(dataset=dataset,
                             model_entity=model)

        ##############
        # Predicting #
        ##############
        logger.info(f"Calling prediction, dataset: {dataset.name!r} ({model.id!r}), filters: {filters}")
        if not filters:
            filters = entities.Filters()
        self.predict_dataset(dataset=dataset,
                             filters=filters,
                             with_upload=True)

        ##############
        # Evaluating #
        ##############
        logger.info(f"Starting adapter.evaluate()")
        if progress is not None:
            progress.update(message='calculating metrics',
                            progress=98)
        model = self.evaluate(model=model,
                              dataset=dataset,
                              filters=filters)
        #########
        # Done! #
        #########
        if progress is not None:
            progress.update(message='finishing evaluation',
                            progress=99)
        return model, dataset

    # =============
    # INNER METHODS
    # =============

    @staticmethod
    def _upload_model_features(feature_set_id, project_id, item: entities.Item, vector):
        try:
            if vector is not None:
                item.features.create(value=vector,
                                     project_id=project_id,
                                     feature_set_id=feature_set_id,
                                     entity=item)
        except Exception as e:
            logger.error(f'Failed to upload feature vector of length {len(vector)} to item {item.id}, Error: {e}')

    def _upload_model_annotations(self, item: entities.Item, predictions, clean_annotations):
        """
        Utility function that upload prediction to dlp platform based on the package.output_type
        :param predictions: `dl.AnnotationCollection`
        :param cleanup: `bool` if set removes existing predictions with the same package-model name
        """
        if not (isinstance(predictions, entities.AnnotationCollection) or isinstance(predictions, list)):
            raise TypeError('predictions was expected to be of type {}, but instead it is {}'.
                            format(entities.AnnotationCollection, type(predictions)))
        if clean_annotations:
            clean_filter = entities.Filters(resource=entities.FiltersResource.ANNOTATION)
            clean_filter.add(field='metadata.user.model.name', values=self.model_entity.name)
            # clean_filter.add(field='type', values=self.model_entity.output_type,)
            item.annotations.delete(filters=clean_filter)
        annotations = item.annotations.upload(annotations=predictions)
        return annotations

    @staticmethod
    def _item_to_image(item):
        """
        Preprocess items before calling the `predict` functions.
        Convert item to numpy array

        :param item:
        :return:
        """
        buffer = item.download(save_locally=False)
        image = np.asarray(Image.open(buffer))
        return image

    @staticmethod
    def _item_to_item(item):
        """
        Default item to batch function.
        This function should prepare a single item for the predict function, e.g. for images, it loads the image as numpy array
        :param item:
        :return:
        """
        return item

    @staticmethod
    def _item_to_text(item):
        filename = item.download(overwrite=True)
        text = None
        if item.mimetype == 'text/plain' or item.mimetype == 'text/markdown':
            with open(filename, 'r') as f:
                text = f.read()
                text = text.replace('\n', ' ')
        else:
            logger.warning('Item is not text file. mimetype: {}'.format(item.mimetype))
            text = item
        if os.path.exists(filename):
            os.remove(filename)
        return text

    @staticmethod
    def _uri_to_image(data_uri):
        # data_uri = "data:image/png;base64,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"
        image_b64 = data_uri.split(",")[1]
        binary = base64.b64decode(image_b64)
        image = np.asarray(Image.open(io.BytesIO(binary)))
        return image

    def _update_predictions_metadata(self, item: entities.Item, predictions: entities.AnnotationCollection):
        """
        add model_name and model_id to the metadata of the annotations.
        add model_info to the metadata of the system metadata of the annotation.
        Add item id to all the annotations in the AnnotationCollection

        :param item: Entity.Item
        :param predictions: item's AnnotationCollection
        :return:
        """
        for prediction in predictions:
            if prediction.type == entities.AnnotationType.SEGMENTATION:
                color = None
                try:
                    color = item.dataset._get_ontology().color_map.get(prediction.label, None)
                except (exceptions.BadRequest, exceptions.NotFound):
                    ...
                if color is None:
                    if self.model_entity._dataset is not None:
                        try:
                            color = self.model_entity.dataset._get_ontology().color_map.get(prediction.label,
                                                                                            (255, 255, 255))
                        except (exceptions.BadRequest, exceptions.NotFound):
                            ...
                if color is None:
                    logger.warning("Can't get annotation color from model's dataset, using default.")
                    color = prediction.color
                prediction.color = color

            prediction.item_id = item.id
            if 'user' in prediction.metadata and 'model' in prediction.metadata['user']:
                prediction.metadata['user']['model']['model_id'] = self.model_entity.id
                prediction.metadata['user']['model']['name'] = self.model_entity.name
            if 'system' not in prediction.metadata:
                prediction.metadata['system'] = dict()
            if 'model' not in prediction.metadata['system']:
                prediction.metadata['system']['model'] = dict()
            confidence = prediction.metadata.get('user', dict()).get('model', dict()).get('confidence', None)
            prediction.metadata['system']['model'] = {
                'model_id': self.model_entity.id,
                'name': self.model_entity.name,
                'confidence': confidence
            }

    ##############################
    # Callback Factory functions #
    ##############################
    @property
    def dataloop_keras_callback(self):
        """
        Returns the constructor for a keras api dump callback
        The callback is used for dlp platform to show train losses

        :return: DumpHistoryCallback constructor
        """
        try:
            import keras
        except (ImportError, ModuleNotFoundError) as err:
            raise RuntimeError(
                '{} depends on extenral package. Please install '.format(self.__class__.__name__)) from err

        import os
        import time
        import json

        class DumpHistoryCallback(keras.callbacks.Callback):
            def __init__(self, dump_path):
                super().__init__()
                if os.path.isdir(dump_path):
                    dump_path = os.path.join(dump_path,
                                             '__view__training-history__{}.json'.format(time.strftime("%F-%X")))
                self.dump_file = dump_path
                self.data = dict()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for name, val in logs.items():
                    if name not in self.data:
                        self.data[name] = {'x': list(), 'y': list()}
                    self.data[name]['x'].append(float(epoch))
                    self.data[name]['y'].append(float(val))
                self.dump_history()

            def dump_history(self):
                _json = {
                    "query": {},
                    "datasetId": "",
                    "xlabel": "epoch",
                    "title": "training loss",
                    "ylabel": "val",
                    "type": "metric",
                    "data": [{"name": name,
                              "x": values['x'],
                              "y": values['y']} for name, values in self.data.items()]
                }

                with open(self.dump_file, 'w') as f:
                    json.dump(_json, f, indent=2)

        return DumpHistoryCallback


class ModelAdapter(dl.BaseModelAdapter):
    """
    Specific Model adapter.
    The class bind a dl.Model entity with the package code
    """
    # TODO:
    #   1) docstring for your ModelAdapter
    #   2) implement the virtual methods for full adapter support
    #   3) add your _defaults

    _defaults = {}

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        raise NotImplementedError("Please implement 'load' method in {}".format(self.__class__.__name__))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        raise NotImplementedError("Please implement 'save' method in {}".format(self.__class__.__name__))

    def train(self, data_path, dump_path, **kwargs):
        """
        Virtual method - need to implement
        Train the model according to data in local_path and save everything to dump_path

        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param dump_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        raise NotImplementedError("Please implement 'train' method in {}".format(self.__class__.__name__))

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        raise NotImplementedError("Please implement 'predict' method in {}".format(self.__class__.__name__))

    def convert(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """
        raise NotImplementedError("Please implement 'convert' method in {}".format(self.__class__.__name__))

    # NOT IN USE
    def convert_dlp(self, items: dl.entities.PagedEntities):
        """ This should implement similar to convert only to work on dlp items.  ->
                   -> meaning create the converted version from items entities"""
        # TODO
        pass


