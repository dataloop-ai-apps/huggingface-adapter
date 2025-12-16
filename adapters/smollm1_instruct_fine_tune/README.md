# SmolLM Model Adapter for Fine-Tuning on Dataloop  

This repository provides a **QLoRA-based fine-tuning adapter** for **SmolLM**, enabling efficient fine-tuning with **4-bit quantization**.  

This adapter is designed for use within the **Dataloop AI platform** and can be **installed and utilized for both fine-tuning and inference** directly from the **Dataloop Model Marketplace**.  

---

## 🚀 Features  

- **Fine-Tuning Support**: Easily fine-tune **SmolLM** using your own datasets on **Dataloop**.  
- **Model Marketplace Integration**: Install and use the adapter from the **Dataloop Model Marketplace**.  
- **Flexible Model Loading**: Supports models from **local paths** or **HuggingFace's Model Hub**.  
- **Automatic Tokenizer Setup**: Configures necessary **tokens and chat templates**.  
- **Efficient Training**: Uses **4-bit quantization (QLoRA)** for **resource-efficient fine-tuning**.  
- **Optimized Inference**: Supports **bfloat16 precision** for **fast GPU inference**.  
- **Lightweight Models**: SmolLM models are compact and efficient, ideal for edge deployment.

---

## 📌 Requirements  

Ensure the following dependencies are installed:  

- [Dataloop SDK](https://github.com/dataloop-ai/dtlpy)  
- To install the necessary dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

## 📂 Dataset Structure

This adapter supports **Dataloop Prompt Items** for fine-tuning. Each prompt item contains a user message with the assistant response as an annotation.

### Using Dataloop Prompt Items

Upload prompt items to your Dataloop dataset with:
- **User message**: As the prompt content
- **Assistant response**: As a text annotation
- **Tags**: `train` or `validation` to split the dataset

### Alternative: JSON Format

You can also use JSON files with the following structure:

```json
[
  {
    "messages": [
      {"role": "user", "content": "My smart light bulb isn't responding. What should I do?"},
      {"role": "assistant", "content": "First, try turning the bulb off and on manually. If that doesn't work, reset the bulb by turning it on and off five times, then reconnect it to the app."}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "My security camera feed is lagging a lot. Any fixes?"},
      {"role": "assistant", "content": "Check if other devices are using a lot of bandwidth. If possible, switch your camera to a 5GHz Wi-Fi network for a faster connection."}
    ]
  }
]
```

💡 **Notes**: 
* To finetune the model, you need to provide both `train` and `validation` data tagged appropriately in your Dataloop dataset.
* Currently, this adapter supports only **chatbot-style fine-tuning**.

## 🔧 Model Configuration

The model configuration is defined in `dataloop.json`. Below is an explanation of each configuration parameter:

### Model Settings
- **system_prompt**: Sets the initial prompt for the model, defining its behavior and tone. (Default: `"You are a helpful and a bit cynical assistant. Give relevant and short answers, if you don't know the answer just say it, don't make up an answer"`)
- **model_name**: The HuggingFace model to use. (Default: `"HuggingFaceTB/SmolLM-1.7B-Instruct"`). Other options:
  - `"HuggingFaceTB/SmolLM-360M-Instruct"` - Smallest, fastest
  - `"HuggingFaceTB/SmolLM-1.7B-Instruct"` - Balanced performance

### LoRA Parameters
- **r**: The rank parameter for LoRA, controlling the number of low-rank matrices. (Default: `8`)
- **lora_alpha**: A scaling factor for LoRA, affecting the learning rate of the low-rank matrices. (Default: `16`)
- **lora_dropout**: Dropout rate applied to LoRA layers to prevent overfitting. (Default: `0.05`)
- **task_type**: The type of task. (Default: `"CAUSAL_LM"`)
- **target_modules**: Model modules to apply LoRA. (Default: `["q_proj", "v_proj", "k_proj", "o_proj"]`)

### Training Parameters
- **num_train_epochs**: Number of training epochs. (Default: `15`)
- **per_device_train_batch_size**: Batch size per device during training. (Default: `1`)
- **gradient_accumulation_steps**: Steps to accumulate gradients before updating. (Default: `16`)
- **optim**: Optimizer to use. (Default: `"paged_adamw_32bit"`)
- **save_steps**: Steps between checkpoint saves. (Default: `10`)
- **logging_steps**: Steps between logging updates. (Default: `10`)
- **learning_rate**: Initial learning rate. (Default: `2e-4`)
- **warmup_ratio**: Ratio of steps for learning rate warmup. (Default: `0.03`)
- **lr_scheduler_type**: Learning rate scheduler type. (Default: `"constant"`)
- **save_every_n_epochs**: Epochs between checkpoint saves. (Default: `2`)

### Compute Resources
- **Default GPU**: `gpu-t4` is used for training by default
- **Changing GPU**: You can change the machine type from the Dataloop UI during the training phase:
  - `gpu-t4` - Default, good for most fine-tuning tasks
  - `gpu-t4-m` - More memory
  - `gpu-a100` - Recommended for larger models or faster training

### Advanced Training Settings
- **bf16**: Use bfloat16 precision. (Default: `true`)
- **group_by_length**: Group sequences by length for efficiency. (Default: `true`)
- **save_total_limit**: Maximum checkpoints to keep. (Default: `3`)
- **max_grad_norm**: Maximum gradient norm for clipping. (Default: `0.3`)
- **remove_unused_columns**: Remove unused dataset columns. (Default: `false`)
- **gradient_checkpointing**: Use gradient checkpointing to save memory. (Default: `true`)
- **use_reentrant**: Use reentrant gradient checkpointing. (Default: `false`)

### Logging Settings
- **report_to**: Platforms to report metrics. (Default: `["tensorboard"]`)
- **logging_first_step**: Log the first training step. (Default: `true`)
- **log_level**: Logging level. (Default: `"info"`)
- **logging_strategy**: Logging strategy. (Default: `"steps"`)

### Generation Parameters (Inference)
- **max_new_tokens**: Maximum tokens to generate. (Default: `512`)
- **temperature**: Controls randomness - higher = more random. (Default: `0.7`)
- **do_sample**: Use sampling instead of greedy decoding. (Default: `true`)
- **top_p**: Nucleus sampling parameter. (Default: `0.95`)
- **repetition_penalty**: Penalty for repeating tokens. (Default: `1.1`)

## 🔧 Installation

Install the model from [Dataloop's Marketplace](https://dataloop.ai/platform/marketplace/).

## 📜 License

This project uses the SmolLM model which is released under the Apache 2.0 License.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, feel free to open an issue or submit a pull request.
