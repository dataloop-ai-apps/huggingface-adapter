import unittest
import os
import json
from typing import List
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import dtlpy as dl

# Import the HuggingAdapter class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate.responses.pegasus-xsum_2025_03_19_15_08_56 import HuggingAdapter


class TestPegasusAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set up common test resources
        cls.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.tests_path = os.path.join(cls.root_path, 'tests', 'example_data')
        
    def setUp(self) -> None:
        # Set up a fresh instance for each test
        self.adapter = HuggingAdapter()
        self.adapter.model_entity = MagicMock()
        self.adapter.model_entity.id = "test-model-id"
        self.adapter.configuration = {
            "model_name": "pegasus-test",
            "device": "cpu"
        }
        
        # Mock tokenizer and model
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        
        # Setup model to return a tensor that the tokenizer can decode
        self.mock_output = torch.tensor([[1, 2, 3]])
        self.mock_model.generate.return_value = self.mock_output
        self.mock_tokenizer.decode.return_value = "This is a test summary."
        
        # Patch the transformers imports
        patch_tokenizer = patch('generate.responses.pegasus-xsum_2025_03_19_15_08_56.PegasusTokenizer')
        patch_model = patch('generate.responses.pegasus-xsum_2025_03_19_15_08_56.PegasusForConditionalGeneration')
        
        self.mock_tokenizer_class = patch_tokenizer.start()
        self.mock_model_class = patch_model.start()
        
        self.mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        self.mock_model_class.from_pretrained.return_value = self.mock_model
        
        self.addCleanup(patch_tokenizer.stop)
        self.addCleanup(patch_model.stop)
        
        # Load the model
        self.adapter.load("dummy/path")

    def test_prepare_item_func_valid_text_item(self):
        """Test prepare_item_func with a valid text item"""
        # Create a mock text item
        mock_item = MagicMock(spec=dl.Item)
        mock_item.mimetype = "text/plain"
        
        # Create a mock prompt item that should be returned
        mock_prompt_item = MagicMock(spec=dl.PromptItem)
        
        # Mock the from_item method
        with patch('dtlpy.PromptItem.from_item', return_value=mock_prompt_item) as mock_from_item:
            result = self.adapter.prepare_item_func(mock_item)
            
            # Verify the from_item was called with the mock item
            mock_from_item.assert_called_once_with(mock_item)
            
            # Verify the result is the mock prompt item
            self.assertEqual(result, mock_prompt_item)
    
    def test_prepare_item_func_valid_json_item(self):
        """Test prepare_item_func with a valid JSON item"""
        # Create a mock JSON item
        mock_item = MagicMock(spec=dl.Item)
        mock_item.mimetype = "application/json"
        
        # Create a mock prompt item that should be returned
        mock_prompt_item = MagicMock(spec=dl.PromptItem)
        
        # Mock the from_item method
        with patch('dtlpy.PromptItem.from_item', return_value=mock_prompt_item) as mock_from_item:
            result = self.adapter.prepare_item_func(mock_item)
            
            # Verify the from_item was called with the mock item
            mock_from_item.assert_called_once_with(mock_item)
            
            # Verify the result is the mock prompt item
            self.assertEqual(result, mock_prompt_item)
    
    def test_prepare_item_func_invalid_item(self):
        """Test prepare_item_func with an invalid item type"""
        # Create a mock item with an invalid mimetype
        mock_item = MagicMock(spec=dl.Item)
        mock_item.mimetype = "image/jpeg"
        
        # Verify that ValueError is raised
        with self.assertRaises(ValueError) as context:
            self.adapter.prepare_item_func(mock_item)
        
        # Check the error message
        self.assertTrue("Item must be of type 'text' or 'json'" in str(context.exception))
    
    def test_predict_valid_prompt(self):
        """Test predict with a valid prompt"""
        # Create a mock prompt item with content
        mock_prompt_item = MagicMock(spec=dl.PromptItem)
        mock_prompt_item.content = {"value": "This is a test text to summarize."}
        
        # Set up the tokenizer return value
        tokenizer_return = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.mock_tokenizer.return_value = tokenizer_return
        
        # Execute predict
        result = self.adapter.predict([mock_prompt_item])
        
        # Verify tokenizer was called with the right text
        self.mock_tokenizer.assert_called_once_with(
            "This is a test text to summarize.", 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )
        
        # Verify model.generate was called
        self.mock_model.generate.assert_called_once()
        
        # Verify tokenizer.decode was called with the output
        self.mock_tokenizer.decode.assert_called_once_with(self.mock_output[0], skip_special_tokens=True)
        
        # Verify prompt_item.add was called with the right parameters
        mock_prompt_item.add.assert_called_once_with(
            message={
                "role": "assistant",
                "content": [{"mimetype": dl.PromptType.TEXT, "value": "This is a test summary."}],
            },
            model_info={
                "name": "pegasus-test",
                "confidence": 1.0,
                "model_id": "test-model-id",
            },
        )
        
        # Verify the function returns an empty list
        self.assertEqual(result, [])
    
    def test_predict_empty_prompt(self):
        """Test predict with an empty prompt"""
        # Create a mock prompt item with empty content
        mock_prompt_item = MagicMock(spec=dl.PromptItem)
        mock_prompt_item.content = {"value": ""}
        
        # Verify that ValueError is raised
        with self.assertRaises(ValueError) as context:
            self.adapter.predict([mock_prompt_item])
        
        # Check the error message
        self.assertTrue("No text found in prompt item" in str(context.exception))
    
    def test_predict_multiple_items(self):
        """Test predict with multiple prompt items"""
        # Create mock prompt items
        mock_prompt_item1 = MagicMock(spec=dl.PromptItem)
        mock_prompt_item1.content = {"value": "Text 1 to summarize."}
        
        mock_prompt_item2 = MagicMock(spec=dl.PromptItem)
        mock_prompt_item2.content = {"value": "Text 2 to summarize."}
        
        # Set up the tokenizer return value
        tokenizer_return = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.mock_tokenizer.return_value = tokenizer_return
        
        # Execute predict
        result = self.adapter.predict([mock_prompt_item1, mock_prompt_item2])
        
        # Verify tokenizer was called twice with different texts
        expected_calls = [
            unittest.mock.call("Text 1 to summarize.", return_tensors="pt", truncation=True, padding=True),
            unittest.mock.call("Text 2 to summarize.", return_tensors="pt", truncation=True, padding=True)
        ]
        self.assertEqual(self.mock_tokenizer.call_count, 2)
        
        # Verify model.generate was called twice
        self.assertEqual(self.mock_model.generate.call_count, 2)
        
        # Verify tokenizer.decode was called twice
        self.assertEqual(self.mock_tokenizer.decode.call_count, 2)
        
        # Verify both prompt items had add called
        mock_prompt_item1.add.assert_called_once()
        mock_prompt_item2.add.assert_called_once()
        
        # Verify the function returns an empty list
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()