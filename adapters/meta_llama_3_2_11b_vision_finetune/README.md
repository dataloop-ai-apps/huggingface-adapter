# LLaMA 3.2 Vision/Vision Instruct Model Adapter for Fine-Tuning on Dataloop

This repository provides a **LoRA-based fine-tuning adapter** for **LLaMA 3.2 Vision/Vision Instruct**, enabling efficient fine-tuning with **LoRA** and other parameter-efficient methods.

This adapter is designed for use within the **Dataloop AI platform** and can be **installed and utilized for both fine-tuning and inference** directly from the **Dataloop Model Marketplace**.

---

## Features

- **Multimodal Understanding**: Process both images and text inputs to generate coherent text responses
- **Fine-Tuning Support**: Easily fine-tune **LLaMA 3.2 Vision/Vision Instruct** using your own datasets on **Dataloop**
- **Multiple Training Strategies**: Support for LoRA, full fine-tuning, and selective freezing (vision encoder or language model)
- **GPU Optimized**: Uses bfloat16 precision and device mapping for efficient training and inference
- **Automatic Chat Template**: Configures necessary **chat templates and special tokens** for LLaMA 3.2 Vision
- **Customizable Parameters**: Fine-tune the model with adjustable hyperparameters
- **Training Monitoring**: Real-time loss tracking and model checkpointing

---

## Requirements

- Dataloop account with appropriate GPU access (GPU-A100-S recommended)
- Hugging Face API key with access to LLaMA 3.2 Vision models
- Dataset containing image-text pairs in the proper format

## Getting Started

1. Add the `dl-huggingface-api-key` integration to your project
2. Install this DPK in your Dataloop project
3. Create a dataset with your training data in prompt item format
4. Deploy the model using the Dataloop platform

## Dataset Format

Your dataset should contain training examples in prompt items with:

- Images as the original items
- Prompt items with structured messages containing both image and text
- Free-text annotations as the expected model responses

Prompt item JSONs should be formatted as follows:

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
          "mimetype": "image/*",
          "value": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
          }
      ]
    },
    {
      "role": "assistant",
      "content": "This is the expected response describing the image."
    }
  ]
} 
```

**Important Notes:**

- Images should be uploaded as original items in your dataset
- Prompt items should reference images via their stream URLs
- Training and validation subsets should be tagged with "train" and "validation" respectively
- Free-text annotations on prompt items serve as the target responses for training

## Fine-Tuning

To fine-tune the model:

1. Create training and validation subsets in your dataset (tagged with "train" and "validation")
2. Configure hyperparameters in the model configuration
3. Run the training process from the Dataloop UI or SDK
4. Monitor training progress through the built-in metrics and callbacks

## Inference

For inference, use the `predict_items` function with prompt items containing image and text inputs to get the model's responses.

The model automatically:

- Processes images and text using the LLaMA 3.2 Vision chat template
- Handles base64-encoded images from prompt items
- Generates responses with configurable parameters
- Manages GPU memory efficiently

## Parameters

The model supports various hyperparameters including:

### Model Configuration

- `model_name`: HuggingFace model name (default: "meta-llama/Llama-3.2-11B-Vision")
- `device_map`: Device mapping strategy (default: "auto")
- `prompt_items_dir`: Directory for prompt items (default: "/prompt_items")
- `system_prompt`: Optional system prompt for inference

### Training Parameters

- `num_train_epochs`: Number of training epochs (default: 10)
- `save_every_n_epochs`: Save model every N epochs (default: 1)
- `per_device_train_batch_size`: Batch size per device during training (default: 5)
- `gradient_accumulation_steps`: Number of gradient accumulation steps (default: 16)
- `warmup_steps`: Number of warmup steps (default: 2)
- `learning_rate`: Learning rate for training (default: 2e-4)
- `weight_decay`: Weight decay (default: 1e-6)
- `adam_beta2`: Beta2 parameter for Adam optimizer (default: 0.999)
- `optim`: Optimizer type (default: "paged_adamw_32bit")
- `bf16`: Whether to use bfloat16 precision (default: true)
- `save_strategy`: Model saving strategy (default: "epoch")
- `logging_strategy`: Logging strategy (default: "epoch")
- `evaluation_strategy`: Evaluation strategy (default: "epoch")
- `remove_unused_columns`: Remove unused columns (default: false)
- `dataloader_pin_memory`: Pin memory for dataloader (default: false)

### LoRA Parameters

- `use_lora`: Whether to use LoRA for fine-tuning (default: true)
- `r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA alpha parameter (default: 8)
- `lora_dropout`: LoRA dropout rate (default: 0.1)
- `use_dora`: Whether to use DoRA (default: true)
- `init_lora_weights`: LoRA weights initialization method (default: "gaussian")
- `target_modules`: Target modules for LoRA (default: ["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"])

### Alternative Training Strategies

- `freeze_llm`: Freeze the language model parameters (default: false)
- `freeze_image`: Freeze the vision encoder parameters (default: false)

### Generation Parameters

- `max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.7)
- `do_sample`: Whether to use sampling (default: true)
- `use_cache`: Whether to use KV cache (default: true)
- `top_p`: Top-p sampling parameter (default: 0.9)

## Example Usage with SDK

```python
import dtlpy as dl

# Login to Dataloop
dl.login()

# Get project and model
project = dl.projects.get('<your-project-id>')
model = project.models.get('llama-vision-finetune')

# Run inference on a prompt item
item = dataset.items.get('item-id')
model.predict_items(items=[item])

# Train the model
model.train()
```

## Training Monitoring

The adapter includes built-in monitoring features:

- Real-time training and validation loss tracking
- Automatic model checkpointing (saves best model based on loss)
- Training logs saved to JSON format
- Progress callbacks for FaaS environments
- GPU memory management and optimization

## Model Architecture

This adapter supports multiple training strategies:

1. **LoRA Fine-tuning** (recommended): Efficient parameter-efficient fine-tuning
2. **Full Fine-tuning**: Complete model parameter updates
3. **Selective Freezing**: Freeze either vision encoder or language model
4. **DoRA Integration**: Enhanced LoRA with dynamic rank adaptation

The model automatically handles:

- LLaMA 3.2 Vision chat template formatting
- Image processing and tokenization
- Memory-efficient training with gradient accumulation
- Automatic device mapping for multi-GPU setups
