# Llama 3.2 Vision Instruct Fine-Tuning

This package provides the Llama 3.2 11B Vision Instruct model from Meta, ready for fine-tuning with QLoRA on Dataloop.

## Features

- **Multimodal Understanding**: Process both images and text inputs to generate coherent text responses
- **QLoRA Fine-Tuning**: Efficiently fine-tune the model on your custom dataset with parameter-efficient methods
- **GPU Optimized**: Uses 4-bit quantization and other optimizations for efficient training and inference
- **Customizable Parameters**: Fine-tune the model with adjustable hyperparameters

## Requirements

- Dataloop account with appropriate GPU access (A100 recommended)
- Hugging Face API key with access to Llama 3.2 models
- Dataset containing image-text pairs in the proper format

## Getting Started

1. Add the `dl-huggingface-api-key` integration to your project
2. Install this DPK in your Dataloop project
3. Create a dataset with your training data
4. Deploy the model using the Dataloop platform

## Dataset Format

Your dataset should contain training examples with:

- Images (for visual input)
- Conversation format JSON files with proper structure:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Your system prompt here"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "value": "path/to/image.jpg"
        },
        {
          "type": "text",
          "value": "What can you see in this image?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "This is the expected model output describing the image."
    }
  ]
}
```

## Fine-Tuning

To fine-tune the model:

1. Create training and validation subsets in your dataset
2. Configure hyperparameters in the model configuration
3. Run the `train_model` function from the Dataloop UI or SDK

## Inference

For inference, use the `predict_items` function with image inputs and prompt text to get the model's responses.

## Parameters

The model supports various hyperparameters including:
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for training
- `r` and `lora_alpha`: LoRA rank and alpha parameters
- Generation parameters like `temperature`, `top_p`, etc.

## Example Usage with SDK

```python
import dtlpy as dl

# Login to Dataloop
dl.login()

# Get project and model
project = dl.projects.get('your-project-id')
model = project.models.get('meta-llama/Llama-3.2-11B-Vision-Instruct')

# Run inference on an item
item = dataset.items.get('item-id')
model.predict_items(items=[item])
``` 