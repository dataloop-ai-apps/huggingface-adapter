# LLaMA 3.2 Text Model Adapter for Fine-Tuning on Dataloop  

This repository provides a **QLoRA-based fine-tuning adapter** for **LLaMA 3.2**, enabling efficient fine-tuning with **4-bit quantization and CPU offloading**.  

This adapter is designed for use within the **Dataloop AI platform** and can be **installed and utilized for both fine-tuning and inference** directly from the **Dataloop Model Marketplace**.  

---

## 🚀 Features  

- **Fine-Tuning Support**: Easily fine-tune **LLaMA 3.2** using your own datasets on **Dataloop**.  
- **Model Marketplace Integration**: Install and use the adapter from the **Dataloop Model Marketplace**.  
- **Flexible Model Loading**: Supports models from **local paths** or **HuggingFace's Model Hub**.  
- **Automatic Tokenizer Setup**: Configures necessary **tokens and chat templates**.  
- **Efficient Training**: Uses **4-bit quantization** for **resource-efficient fine-tuning**.  
- **Optimized Inference**: Supports **bfloat16 precision** for **fast GPU inference**.  

---

## 📌 Requirements  

Ensure the following dependencies are installed:  

- [Dataloop SDK](https://github.com/dataloop-ai/dtlpy)  
- [HuggingFace Token](https://github.com/huggingface/transformers)  
- To install the necessary dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

## 📂 Dataset Structure
This adapter currently supports chatbot-style fine-tuning datasets formatted as JSON. Each dataset should contain a list of conversations, where each example includes a "messages" key mapping to a list of messages with "role" and "content".

### Example Chat Dataset
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a customer support assistant specializing in smart home devices."},
      {"role": "user", "content": "My smart light bulb isn't responding. What should I do?"},
      {"role": "assistant", "content": "First, try turning the bulb off and on manually. If that doesn't work, reset the bulb by turning it on and off five times, then reconnect it to the app."}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "You assist users in troubleshooting smart home security cameras."},
      {"role": "user", "content": "My security camera feed is lagging a lot. Any fixes?"},
      {"role": "assistant", "content": "Check if other devices are using a lot of bandwidth. If possible, switch your camera to a 5GHz Wi-Fi network for a faster connection."}
    ]
  }
]
```
💡 Note: Currently, this adapter supports only chatbot-style fine-tuning. If you need support for other dataset formats, please contact us.

## 🔧 Model Configuration

The model configuration is defined in a JSON file, typically `dataloop.json`. Below is an explanation of each configuration parameter along with their default values:

- **system_prompt**: Sets the initial prompt for the model, defining its behavior and tone. (Default: `"You are a helpful and a bit cynical assistant. Give relevant and short answers, if you don't know the answer just say it, don't make up an answer"`)
- **model_name**: The name of the model to be used. (Default: `"meta-llama/Llama-3.1-8B-Instruct"`)
- **r**: The rank parameter for LoRA, controlling the number of low-rank matrices. (Default: `16`)
- **lora_alpha**: A scaling factor for LoRA, affecting the learning rate of the low-rank matrices. (Default: `32`)
- **lora_dropout**: Dropout rate applied to LoRA layers to prevent overfitting. (Default: `0.05`)
- **task_type**: The type of task, e.g., "CAUSAL_LM" for causal language modeling. (Default: `"CAUSAL_LM"`)
- **target_modules**: A list of model modules to which LoRA should be applied. (Default: `[]`)
- **num_train_epochs**: Number of training epochs. (Default: `15`)
- **per_device_train_batch_size**: Batch size per device during training. (Default: `1`)
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating model parameters. (Default: `16`)
- **optim**: Optimizer to use, e.g., "paged_adamw_32bit". (Default: `"paged_adamw_32bit"`)
- **save_steps**: Number of steps between model checkpoint saves. (Default: `10`)
- **logging_steps**: Number of steps between logging updates. (Default: `10`)
- **learning_rate**: Initial learning rate for training. (Default: `2e-4`)
- **warmup_ratio**: Ratio of total training steps used for learning rate warmup. (Default: `0.03`)
- **lr_scheduler_type**: Type of learning rate scheduler, e.g., "constant". (Default: `"constant"`)
- **bf16**: Boolean indicating whether to use bfloat16 precision. (Default: `true`)
- **group_by_length**: Boolean indicating whether to group sequences of similar length for efficient training. (Default: `true`)
- **save_total_limit**: Maximum number of checkpoints to keep. (Default: `3`)
- **max_grad_norm**: Maximum gradient norm for gradient clipping. (Default: `0.3`)
- **remove_unused_columns**: Boolean indicating whether to remove unused columns from the dataset. (Default: `false`)
- **gradient_checkpointing**: Boolean indicating whether to use gradient checkpointing to save memory. (Default: `true`)
- **use_reentrant**: Boolean indicating whether to use reentrant gradient checkpointing. (Default: `false`)
- **report_to**: List of platforms to report training metrics, e.g., ["tensorboard"]. (Default: `["tensorboard"]`)
- **logging_first_step**: Boolean indicating whether to log the first training step. (Default: `true`)
- **log_level**: Logging level, e.g., "info". (Default: `"info"`)
- **logging_strategy**: Strategy for logging, e.g., "steps". (Default: `"steps"`)
- **save_every_n_epochs**: Number of epochs between model checkpoint saves. (Default: `2`)

## 🔧 Installation

Install the model from [Dataloop's Marketplace](https://dataloop.ai/platform/marketplace/).


## 📜 License
This project is licensed under the Llama-3.2-1B License. See the LICENSE file for more details.


## 🤝 Contributing
Contributions are welcome! If you have ideas for improvements, feel free to open an issue or submit a pull request.
