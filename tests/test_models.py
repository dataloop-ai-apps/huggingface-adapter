import unittest
import dtlpy as dl
import os
import json
import random
import torch
import numpy as np


SEED = 1337
BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "HF-Models-Tests"


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    root_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapters_path: str = os.path.join(root_path, 'adapters')
    tests_path: str = os.path.join(root_path, 'tests', 'example_data')
    prepare_item_function = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('prod')
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)
        cls.prepare_item_function = {
            "text": cls._prepare_text_item,
            "text_prompt": cls._prepare_text_prompt_item,
            "image_prompt": cls._prepare_image_prompt_item,
            "text_and_image_prompt": "",
        }

    def setUp(self) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        for model in cls.project.models.list().all():
            model.delete()
        dl.logout()

    # Item preparation functions
    def _prepare_text_item(self, item_type: str, model_folder_name: str):
        local_path = os.path.join(self.tests_path, f"{item_type}.txt")
        remote_name = f'{model_folder_name}.txt'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_text_prompt_item(self, item_type: str, model_folder_name: str):
        local_path = os.path.join(self.tests_path, f"{item_type}.json")
        remote_name = f'{model_folder_name}.json'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def _prepare_image_prompt_item(self, item_type: str, model_folder_name: str):
        local_path_image = os.path.join(self.tests_path, f"{item_type}.jpg")
        remote_name_image = f'{model_folder_name}.jpg'
        image_item = self.dataset.items.upload(
            local_path=local_path_image,
            remote_name=remote_name_image,
            overwrite=True
        )

        local_path = os.path.join(self.tests_path, f"{item_type}.json")
        remote_name = f'{model_folder_name}.json'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    # Perdict function
    def _perform_model_predict(self, item_type: str, model_folder_name: str):
        # Upload item
        item = self.prepare_item_function[item_type](item_type=item_type, model_folder_name=model_folder_name)

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
        return model.predict(item_ids=[item.id])

    # Test functions
    def test_amazon_review_sentiment_analysis(self):
        model_folder_name = 'amazon_review_sentiment_analysis'
        item_type = 'text_prompt'
        predict_results = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predict_results, list))  # TODO

    def test_auto_for_causal_lm(self):
        model_folder_name = 'auto_for_causal_lm'
        item_type = 'text_prompt'
        predict_results = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predict_results, list))  # TODO

    def test_bert_base_ner(self):
        model_folder_name = 'bert_base_ner'
        item_type = 'text'
        predict_results = self._perform_model_predict(item_type=item_type, model_folder_name=model_folder_name)
        self.assertTrue(isinstance(predict_results, list))  # TODO


if __name__ == '__main__':
    unittest.main()
