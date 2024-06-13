import unittest
import dtlpy as dl
import os
import json
import random
import torch
import numpy as np
import enum


SEED = 1337
BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
API_KEY = os.environ['API_KEY']
DATASET_NAME = "HF-Models-Tests"


class ItemTypes(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT_PROMPT = "text_prompt"
    IMAGE_PROMPT = "image_prompt"
    TEXT_AND_IMAGE_PROMPT = "text_and_image_prompt"


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    root_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapters_path: str = os.path.join(root_path, 'adapters')
    tests_path: str = os.path.join(root_path, 'tests', 'example_data')
    prepare_item_function = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.root_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)
        cls.prepare_item_function = {
            ItemTypes.TEXT.value: cls._prepare_text_item,
            ItemTypes.IMAGE.value: cls._prepare_image_item,
            ItemTypes.AUDIO.value: cls._prepare_audio_item,
            ItemTypes.TEXT_PROMPT.value: cls._prepare_text_prompt_item,
            ItemTypes.IMAGE_PROMPT.value: cls._prepare_image_prompt_item,
            ItemTypes.TEXT_AND_IMAGE_PROMPT.value: cls._prepare_text_and_image_prompt_item,
        }

    def setUp(self) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all models
        for model in cls.project.models.list().all():
            model.delete()

        # Delete all apps
        for app in cls.project.apps.list().all():
            if app.project.id == cls.project.id:
                app.uninstall()

        # Delete all dpks
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field="scope", values="project")
        for dpk in cls.project.dpks.list(filters=filters).all():
            if dpk.project.id == cls.project.id and dpk.creator == BOT_EMAIL:
                dpk.delete()
        dl.logout()

    # Item preparation functions
    def _prepare_text_item(self, model_folder_name: str):
        item_name = f'{ItemTypes.TEXT.value}.txt'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.txt'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_image_item(self, model_folder_name: str):
        item_name = f'{ItemTypes.IMAGE.value}.jpeg'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.jpeg'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_audio_item(self, model_folder_name: str):
        item_name = f'{ItemTypes.AUDIO.value}.flac'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.flac'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_text_prompt_item(self, model_folder_name: str):
        item_name = f'{ItemTypes.TEXT_PROMPT.value}.json'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.json'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_image_prompt_item(self, model_folder_name: str):
        # Upload image
        item_name_image = f'{ItemTypes.IMAGE.value}.jpeg'
        local_path_image = os.path.join(self.tests_path, item_name_image)
        # remote_name_image = f'{model_folder_name}.jpeg'
        image_item: dl.Item = self.dataset.items.upload(
            local_path=local_path_image,
            # remote_name=remote_name_image,
            # overwrite=True
        )

        # Prepare and upload json
        item_name = f'{ItemTypes.IMAGE_PROMPT.value}.json'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.json'
        with open(local_path, 'r') as f:
            json_data = json.load(f)
        json_data["prompts"]["prompt1"][0]["value"] = image_item.stream
        with open(local_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_text_and_image_prompt_item(self, model_folder_name: str):
        # Upload image
        item_name_image = f'{ItemTypes.IMAGE.value}.jpeg'
        local_path_image = os.path.join(self.tests_path, item_name_image)
        # remote_name_image = f'{model_folder_name}.jpeg'
        image_item: dl.Item = self.dataset.items.upload(
            local_path=local_path_image,
            # remote_name=remote_name_image,
            # overwrite=True
        )

        # Prepare and upload json
        item_name = f'{ItemTypes.TEXT_AND_IMAGE_PROMPT.value}.json'
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.json'
        with open(local_path, 'r') as f:
            json_data = json.load(f)
        json_data["prompts"]["prompt1"][0]["value"] = image_item.stream
        with open(local_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    # API Key preparation functions
    def _prepare_llama_3_8b_instruct_token(self, model: dl.Model, service: dl.Service, model_folder_name: str):
        secret_name = f'{model_folder_name}_key'.replace("-", "_")
        secret_value = API_KEY

        # TODO: Make sure the bot is a member in the project organization with the right permissions
        # Check if secret exists
        project_org = dl.organizations.get(organization_id=self.project.org["id"])
        org_integrations = project_org.integrations.list()

        # Delete existing secret
        for integration in org_integrations:
            if integration["name"] == secret_name:
                secret_id = integration["id"]
                project_org.integrations.delete(integrations_id=secret_id, sure=True, really=True)
                break

        # Create new secret
        secret = project_org.integrations.create(
            integrations_type=dl.IntegrationType.KEY_VALUE,
            name=secret_name,
            options={"key": secret_name, "value": secret_value}
        )
        service.secrets = [secret.id]
        service.update()

        # Update model configuration
        model = model.project.models.get(model_id=model.id)
        model.configuration["hf_access_token"] = secret_name
        model.update()

    # Perdict function
    def _perform_model_predict(self, item_type: ItemTypes, model_folder_name: str, api_key_function=None):
        # Upload item
        item = self.prepare_item_function[item_type.value](self=self, model_folder_name=model_folder_name)

        # Open dataloop json
        model_path = os.path.join(self.adapters_path, model_folder_name)
        dataloop_json_filepath = os.path.join(model_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        model_name = dataloop_json.get('components', dict()).get('models', list())[0].get("name", None)

        # Publish dpk and install app
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        # Get model and predict
        model = app.project.models.get(model_name=model_name)
        service = model.deploy()

        # Prepare API Key
        if api_key_function is not None:
            api_key_function(model=model, service=service, model_folder_name=model_folder_name)

        model.metadata["system"]["deploy"] = {"services": [service.id]}
        execution = model.predict(item_ids=[item.id])
        execution = execution.wait()

        # Execution output format:
        # [[{"item_id": item_id}, ...], [{"annotation_id": annotation_id}, ...]]
        _, annotations = execution.output
        return annotations

    # Test functions
    def test_amazon_review_sentiment_analysis(self):
        model_folder_name = 'amazon_review_sentiment_analysis'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_auto_for_causal_lm(self):
        model_folder_name = 'auto_for_causal_lm'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_bert_base_ner(self):
        model_folder_name = 'bert_base_ner'
        item_type = ItemTypes.TEXT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_blip_image_captioning_large(self):
        model_folder_name = 'blip_image_captioning_large'
        item_type = ItemTypes.TEXT_AND_IMAGE_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_detr_resnet_50_panoptic(self):
        model_folder_name = 'detr_resnet_50_panoptic'
        item_type = ItemTypes.IMAGE
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_detr_resnet_101(self):
        model_folder_name = 'detr_resnet_101'
        item_type = ItemTypes.IMAGE
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_dialogpt_large(self):
        model_folder_name = 'dialogpt_large'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_instruct_pix2pix(self):
        # Delete previous generated items
        generated_items_folder = "/instruct_pix2pix_results"
        filters = dl.Filters()
        filters.add(field='dir', values=generated_items_folder)
        self.dataset.items.delete(filters=filters)

        model_folder_name = 'instruct_pix2pix'
        item_type = ItemTypes.TEXT_AND_IMAGE_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_ivrit_ai_whisper_large_v2_tuned(self):
        model_folder_name = 'ivrit_ai_whisper_large_v2_tuned'
        item_type = ItemTypes.AUDIO
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_meta_llama_3_8b_instruct(self):
        model_folder_name = 'meta_llama_3_8b_instruct'
        item_type = ItemTypes.TEXT_PROMPT
        api_key_function = self._prepare_llama_3_8b_instruct_token
        predicted_annotations = self._perform_model_predict(
            item_type=item_type,
            model_folder_name=model_folder_name,
            api_key_function=api_key_function
        )
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_open_llama_3b(self):
        model_folder_name = 'open_llama_3b'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_pegasus_summarize(self):
        model_folder_name = 'pegasus_summarize'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_stable_diffusion_v1_5(self):
        # Delete previous generated items
        generated_items_folder = "/stable_diffusion_v1_5_results"
        filters = dl.Filters()
        filters.add(field='dir', values=generated_items_folder)
        self.dataset.items.delete(filters=filters)

        model_folder_name = 'stable_diffusion_v1_5'
        item_type = ItemTypes.TEXT_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_uform_gen2_qwen_500m(self):
        model_folder_name = 'uform_gen2_qwen_500m'
        item_type = ItemTypes.TEXT_AND_IMAGE_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_vilt_b32_finetuned_vqa(self):
        model_folder_name = 'vilt_b32_finetuned_vqa'
        item_type = ItemTypes.TEXT_AND_IMAGE_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)

    def test_vit_gpt2_image_captioning(self):
        model_folder_name = 'vit_gpt2_image_captioning'
        item_type = ItemTypes.IMAGE_PROMPT
        predicted_annotations = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)


if __name__ == '__main__':
    unittest.main()
