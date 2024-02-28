import unittest
import dtlpy as dl
import os
import json
import random
import torch
import numpy as np
from model_adapter import ModelAdapter
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, LlamaTokenizer, LlamaForCausalLM,
                          BertForTokenClassification, BertTokenizerFast, DetrForSegmentation, DetrFeatureExtractor,
                          DetrForObjectDetection)

SEED = 1337
BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    package: dl.Package = None

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('prod')
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)

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
        # cls.package.delete()
        dl.logout()

    def test_bert_base_ner(self):
        with open(r'../adapters/bert_base_ner/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, BertForTokenClassification))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, BertTokenizerFast))
        self.assertTrue('bert-base' in model_adapter.hugging.model.name_or_path.lower())

    def test_detr_resnet_50_panoptic(self):
        with open(r'../adapters/detr_resnet_50_panoptic/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, DetrForSegmentation))
        self.assertTrue(isinstance(model_adapter.hugging.feature_extractor, DetrFeatureExtractor))
        self.assertTrue('detr-resnet-50-panoptic' in model_adapter.hugging.model.name_or_path.lower())

    def test_detr_resnet_101(self):
        with open(r'../adapters/detr_resnet_101/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, DetrForObjectDetection))
        self.assertTrue(isinstance(model_adapter.hugging.feature_extractor, DetrFeatureExtractor))
        self.assertTrue('detr-resnet-101' in model_adapter.hugging.model.name_or_path.lower())

    def test_open_llama_3b(self):
        with open(r'../adapters/open_llama_3b/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model)
        model_adapter.hugging.model.config.seed = SEED
        with open("./test_input.json", "r") as f:
            inp = json.load(f)
        ans = model_adapter.predict([inp])
        self.assertTrue(isinstance(model_adapter.hugging.model, LlamaForCausalLM))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, LlamaTokenizer))
        self.assertTrue('open_llama' in model_adapter.hugging.model.name_or_path.lower())
        self.assertEqual(
            ans[0][0].coordinates,
            "I'm not sure if this is a test or not.\nI'm not sure if this is a test or not. I'm not sure if this is a "
            "test or not. I'm not sure if this is a test or not. I'm not sure if this is a test or not. "
            "I'm not sure if this is a test or not. I'm not sure if this is a test or not. I"
            )
        self.assertAlmostEqual(
            ans[0][0]['metadata']['user']['model']['confidence'],
            0.7106203436851501, 3
            )

    def test_dialogpt(self):
        # model = dialogpt_large.model_creation(self.package)

        with open(r'../adapters/dialogpt_large/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model_entity=model)
        model_adapter.hugging.model.config.seed = SEED
        with open("./test_input.json", "r") as f:
            inp = json.load(f)
        ans = model_adapter.predict([inp])
        self.assertTrue(isinstance(model_adapter.hugging.model, GPT2LMHeadModel))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, GPT2TokenizerFast))
        self.assertTrue('dialogpt' in model_adapter.hugging.model.name_or_path.lower())
        self.assertEqual(
            ans[0][0].coordinates,
            "Nah, it's a test to see if you can handle it"
            )
        self.assertAlmostEqual(
            ans[0][0]['metadata']['user']['model']['confidence'],
            0.3476366400718689, 3
            )

    def test_autocausallm(self):
        with open(r'../adapters/auto_for_causal_lm/dataloop.json', 'r') as f:
            config = json.load(f)
        model = dl.Model.from_json(
            _json=config.get('components', dict()).get('models', list())[0],
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package()
        )
        model_adapter = ModelAdapter(model_entity=model)
        model_adapter.hugging.model.config.seed = SEED
        with open("./test_input.json", "r") as f:
            inp = json.load(f)
        ans = model_adapter.predict([inp])
        self.assertTrue(isinstance(model_adapter.hugging.model, GPT2LMHeadModel))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, GPT2TokenizerFast))
        self.assertTrue('microsoft/dialogpt-large' in model_adapter.hugging.model.name_or_path.lower())
        self.assertEqual(
            ans[0][0].coordinates,
            "Nah, it's a test to see if you can handle it"
            )
        self.assertAlmostEqual(
            ans[0][0]['metadata']['user']['model']['confidence'],
            0.3476366400718689,
            3
        )


if __name__ == '__main__':
    unittest.main()
