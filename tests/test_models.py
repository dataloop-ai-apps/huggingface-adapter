import unittest
import dtlpy as dl
from models import (open_llama, dialogpt_large, dslim_bert_base_ner, facebook_detr_resnet_50_panoptic,
                    facebook_detr_resnet_101)
from creation import package_creation
from model_adapter import ModelAdapter
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, LlamaTokenizer, LlamaForCausalLM,
                          BertForTokenClassification, BertTokenizerFast, DetrForSegmentation, DetrFeatureExtractor,
                          DetrForObjectDetection)

TOKEN = '<INSERT-TOKEN-MECHANISM>'


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if dl.token_expired():
            dl.login_token(TOKEN)
        self.project = dl.projects.create("hugging_face_adapter_tests")
        self.package = package_creation(self.project, "../model_adapter.py")

    def tearDown(self) -> None:
        self.project.delete(sure=True, really=True)
        dl.logout()

    def test_open_llama(self):
        model = open_llama.model_creation(self.package)
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, LlamaForCausalLM))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, LlamaTokenizer))
        self.assertTrue('open_llama' in model_adapter.hugging.model.name_or_path.lower())

    def test_dialogpt(self):
        model = dialogpt_large.model_creation(self.package)
        model_adapter = ModelAdapter(model)
        self.assertTrue(isinstance(model_adapter.hugging.model, GPT2LMHeadModel))
        self.assertTrue(isinstance(model_adapter.hugging.tokenizer, GPT2TokenizerFast))
        self.assertTrue('dialogpt' in model_adapter.hugging.model.name_or_path.lower())

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


if __name__ == '__main__':
    unittest.main()
