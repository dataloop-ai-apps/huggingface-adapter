import unittest
import dtlpy as dl
import json
import random
import torch
import numpy as np
from models import (open_llama, dialogpt_large, dslim_bert_base_ner, facebook_detr_resnet_50_panoptic,
                    facebook_detr_resnet_101)
from creation import package_creation
from model_adapter import ModelAdapter
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, LlamaTokenizer, LlamaForCausalLM,
                          BertForTokenClassification, BertTokenizerFast, DetrForSegmentation, DetrFeatureExtractor,
                          DetrForObjectDetection)

SEED = 1337


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    package: dl.Package = None

    @classmethod
    def setUpClass(cls) -> None:
        if dl.token_expired():
            dl.login()
        try:
            cls.project = dl.projects.create("hugging_face_adapter_tests")
        except dl.exceptions.InternalServerError:
            print("Dummy project already exists")
            cls.project = dl.projects.get("hugging_face_adapter_tests")
        cls.package = package_creation(cls.project, "../model_adapter.py")

    def setUp(self) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete(sure=True, really=True)
        dl.logout()

    def test_bert_base(self):
        model = dslim_bert_base_ner.create_model_entity(self.package)
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, BertForTokenClassification))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, BertTokenizerFast))
        self.assertTrue('bert-base' in model_adapter.hugging.model.name_or_path.lower())

    def test_detr_resnet_50_panoptic(self):
        model = facebook_detr_resnet_50_panoptic.create_model_entity(self.package)
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, DetrForSegmentation))
        self.assertTrue(isinstance(model_adapter.hugging.feature_extractor, DetrFeatureExtractor))
        self.assertTrue('detr-resnet-50-panoptic' in model_adapter.hugging.model.name_or_path.lower())

    def test_detr_resnet_101(self):
        model = facebook_detr_resnet_101.create_model_entity(self.package)
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, DetrForObjectDetection))
        self.assertTrue(isinstance(model_adapter.hugging.feature_extractor, DetrFeatureExtractor))
        self.assertTrue('detr-resnet-101' in model_adapter.hugging.model.name_or_path.lower())

    def test_open_llama(self):
        model = open_llama.model_creation(self.package)
        model_adapter = ModelAdapter(model)
        model_adapter.hugging.model.config.seed = SEED
        with open("./test_input.json", "r") as f:
            inp = json.load(f)
        ans = model_adapter.predict([inp])
        self.assertTrue(isinstance(model_adapter.hugging.model, LlamaForCausalLM))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, LlamaTokenizer))
        self.assertTrue('open_llama' in model_adapter.hugging.model.name_or_path.lower())
        self.assertEqual(
            ans[0][0]['coordinates'],
            "I'm not sure if this is a test or not.\nI'm not sure if this is a test or not. I'm not sure if this is a "
            "test or not. I'm not sure if this is a test or not. I'm not sure if this is a test or not. "
            "I'm not sure if this is a test or not. I'm not sure if this is a test or not. I"
            )
        self.assertAlmostEqual(
            ans[0][0]['metadata']['user']['model']['confidence'],
            0.7106203436851501, 3
            )

    def test_dialogpt(self):
        model = dialogpt_large.model_creation(self.package)
        model_adapter = ModelAdapter(model)
        model_adapter.hugging.model.config.seed = SEED
        with open("./test_input.json", "r") as f:
            inp = json.load(f)
        ans = model_adapter.predict([inp])
        self.assertTrue(isinstance(model_adapter.hugging.model, GPT2LMHeadModel))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, GPT2TokenizerFast))
        self.assertTrue('dialogpt' in model_adapter.hugging.model.name_or_path.lower())
        self.assertEqual(
            ans[0][0]['coordinates'],
            "Nah, it's a test to see if you can handle it"
            )
        self.assertAlmostEqual(
            ans[0][0]['metadata']['user']['model']['confidence'],
            0.3476366400718689, 3
            )


if __name__ == '__main__':
    unittest.main()
