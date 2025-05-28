# LLaMA 3.2 Vision Intruct Model Adapter for Fine-Tuning on Dataloop  

This repository provides a **QLoRA-based fine-tuning adapter** for **LLaMA 3.2**, enabling efficient fine-tuning with **QLoRA**.  

This adapter is designed for use within the **Dataloop AI platform** and can be **installed and utilized for both fine-tuning and inference** directly from the **Dataloop Model Marketplace**.

---

## Features  

- **Multimodal Understanding**: Process both images and text inputs to generate coherent text responses
- **Fine-Tuning Support**: Easily fine-tune **LLaMA 3.2 Vision Instruct** using your own datasets on **Dataloop**.  
- **QLoRA Fine-Tuning**: Efficiently fine-tune the model on your custom dataset with parameter-efficient methods
- **GPU Optimized**: Uses 4-bit quantization and other optimizations for efficient training and inference
- **Automatic Tokenizer Setup**: Configures necessary **tokens and chat templates**.  
- **Customizable Parameters**: Fine-tune the model with adjustable hyperparameters

---

## Requirements

- Dataloop account with appropriate GPU access (GPU-T4 recommended)
- Hugging Face API key with access to Llama 3.2 models
- Dataset containing image-text pairs in the proper format

## Getting Started

1. Add the `dl-huggingface-api-key` integration to your project
2. Install this DPK in your Dataloop project
3. Create a dataset with your training data
4. Deploy the model using the Dataloop platform

## Dataset Format

Your dataset should contain training examples in prompt items with:

- Images as the prompt
- Caption or image description as the free-text annotation

Prompt item JSONS should be formatted as follows:

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
          "type": "text",
          "value": "What can you see in this image?"
        },
        {
          "type": "image",
          "value": "https://url.to/image.jpg"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "This is the expected prepared caption or model output describing the image."
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
