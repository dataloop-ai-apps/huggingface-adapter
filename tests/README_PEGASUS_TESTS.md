# Pegasus Adapter Unit Tests

This folder contains unit tests for the Pegasus-XSum adapter, which is an implementation of Dataloop's `BaseModelAdapter` for the Pegasus summarization model.

## Test Coverage

The test file `test_pegasus_adapter.py` includes tests for two main methods:

1. `prepare_item_func` - Tests validating that:
   - Text items are correctly converted to PromptItems
   - JSON items are correctly converted to PromptItems
   - Invalid item types raise appropriate exceptions

2. `predict` - Tests validating that:
   - Valid prompts are properly tokenized and processed
   - The summarized text is correctly added to the prompt item
   - Empty prompts raise appropriate exceptions
   - Multiple items are processed correctly

## Running the Tests

To run the tests, execute the following command from the project root:

```bash
# Run all tests in the test_pegasus_adapter.py file
python -m unittest tests/test_pegasus_adapter.py

# Run a specific test
python -m unittest tests.test_pegasus_adapter.TestPegasusAdapter.test_predict_valid_prompt
```

## Dependencies

The tests mock the actual model and tokenizer to avoid requiring the full Hugging Face model to be downloaded. However, you still need the following dependencies:

- dtlpy
- torch
- transformers
- unittest
- mock 