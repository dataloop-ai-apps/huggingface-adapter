import dtlpy as dl
import json
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
)
from peft import PeftModel
import torch
import transformers
import logging
import subprocess
import threading
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger("finetune-smollm-qlora")


class ModelAdapter(dl.BaseModelAdapter):


    def load(self, local_path, **kwargs):
        self.adapter_defaults.allow_empty_subset = True

        self.hf_model_name = self.model_entity.configuration.get("model_name", "HuggingFaceTB/SmolLM-1.7B-Instruct")
        self.logger.info(f"Model name: {self.hf_model_name}")
        model_path = self.model_entity.configuration.get("model_path")

        # If no local path is provided, use the model name
        if model_path is None:
            self.logger.info("Preparing model and tokenizer for inference from HuggingFace {}".format(self.hf_model_name))
            self.model_path = self.hf_model_name
        else:
            self.model_path = os.path.join(local_path, model_path)

            if not os.path.exists(self.model_path):
                self.logger.warning(f"Path does not exist: {self.model_path}. Loading from HuggingFace...")
                self.model_path = self.hf_model_name
                    
            self.logger.info("Preparing model and tokenizer for inference from {}".format(self.model_path))

        self.model, self.tokenizer = self.create_model_and_tokenizer(model_path=self.model_path)

    def prepare_model_for_finetune(self, model_path):
        """
        Prepare the SmolLM model and tokenizer for fine-tuning with QLoRA.

        This function loads a 4-bit quantized model with CPU offloading enabled,
        ensuring that the model is ready for k-bit training. It also sets up the
        tokenizer with necessary tokens and configurations.

        Parameters:
        - model_path (str): The path to the local model or the model name from HuggingFace.

        Returns:
        - model (transformers.PreTrainedModel): The prepared model ready for fine-tuning.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.

        Raises:
        - ValueError: If CUDA is not available, indicating a potential issue with the GPU configuration.
        """
        # Load 4-bit quantized model with CPU offloading enabled
        if torch.cuda.is_available() is False:
            raise ValueError("CUDA is not available! Check your GPU configuration.")

        # Clear existing model from memory if it was loaded by load() function
        # This prevents device mismatch errors when reloading with quantization
        if hasattr(self, 'model') and self.model is not None:
            self.logger.info("Clearing existing model from memory before loading with quantization...")
            del self.model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            self.logger.info("Memory cleared.")

        # Load tokenizer if not already loaded (reuse from load() if available)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.logger.info(f"Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            # Set pad token if not present
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Tokenizer loaded successfully.")
        else:
            self.logger.info("Reusing existing tokenizer.")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with 4-bit quantization
        # Note: 4-bit quantization doesn't support CPU offloading by default
        # We need to load everything on GPU, or fallback to non-quantized if GPU memory is insufficient
        self.logger.info(f"Loading model with 4-bit quantization from: {model_path}")
        
        try:
            # Try loading with device_map={"": 0} (explicitly puts everything on GPU 0)
            # This is required for 4-bit quantization - it must fit entirely on GPU
            # Using {"": 0} format ensures proper device index for accelerate
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # Explicitly load everything on GPU 0 (required for 4-bit)
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            self.logger.info("Model loaded successfully with 4-bit quantization on GPU.")
            
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e)
            self.logger.warning(f"Failed to load with 4-bit quantization: {error_msg}")
            
            if "dispatched on the CPU" in error_msg or "CPU or the disk" in error_msg or "OutOfMemoryError" in error_msg:
                self.logger.warning("GPU memory insufficient for 4-bit quantization. Trying without quantization...")
                # Fallback: Load without quantization (will use more memory but allows CPU offloading)
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",  # Can use CPU offloading without quantization
                        trust_remote_code=True,
                        use_cache=False,
                        low_cpu_mem_usage=True,
                    )
                    self.logger.warning("Model loaded WITHOUT quantization. Training will require more memory.")
                    self.logger.warning("Consider using a smaller model or increasing GPU memory for quantization.")
                
                except Exception as e2:
                    self.logger.error(f"Failed to load model even without quantization: {str(e2)}")
                    raise Exception(f"Could not load model. Original error: {error_msg}. Fallback error: {str(e2)}")
            else:
                self.logger.error(f"Failed to load model: {error_msg}")
                raise

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # Ensure all necessary tokens are set
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.bos_token_id = self.tokenizer.bos_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id

        return model

    def create_model_and_tokenizer(self, model_path):
        """
        Load the model and tokenizer by name or from local path.
        SmolLM uses its own chat template, so we'll use the model's default template.
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n<|assistant|>\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"""

        # Load model - only normalize if it's a local path (not HuggingFace model ID)
        if os.path.exists(model_path) or os.path.exists(os.path.dirname(model_path)):
            model_path = os.path.normpath(model_path)
            
        is_peft = self._is_local_peft_adapter(model_path)
        
        if is_peft:
            # 1. Load base model (from adapter config)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # 2. Attach adapter from LOCAL PATH
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                local_files_only=True,
            )
            
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        return model, tokenizer

    @staticmethod
    def _is_local_peft_adapter(model_id_or_path: str) -> bool:
        p = Path(model_id_or_path)

        # Must be an existing directory
        if not p.exists() or not p.is_dir():
            return False

        # PEFT adapters always have this
        return (p / "adapter_config.json").is_file()

    def preprocess_chatbot_dataset(self, dataset, max_length=2048):
        """
        Preprocesses a chat-based dataset for fine-tuning SmolLM.

        - Formats messages using the model's chat template.
        - Tokenizes text with truncation and padding.
        - Masks padding tokens in labels (-100) for loss calculation.

        Args:
            dataset: Hugging Face Dataset object or list of dictionaries.
            max_length: Maximum sequence length (increased for SmolLM-2.7B).

        Returns:
            Tokenized dataset with input_ids, attention_mask, and labels.
        """

        def process_function(examples):
            tokenized_outputs = []

            for example in examples["messages"]:
                # Use the tokenizer's chat template if available, otherwise format manually
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                    # Format using chat template
                    formatted_text = self.tokenizer.apply_chat_template(
                        example, tokenize=False, add_generation_prompt=False
                    )
                else:
                    # Fallback: format manually
                    formatted_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in example])

                # Tokenize conversation
                output = self.tokenizer(
                    formatted_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
                )

                # Remove batch dimension
                input_ids = output["input_ids"].squeeze(0)
                attention_mask = output["attention_mask"].squeeze(0)

                # Create labels (mask padding tokens with -100)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                tokenized_outputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

            return {
                "input_ids": torch.stack([x["input_ids"] for x in tokenized_outputs]),
                "attention_mask": torch.stack([x["attention_mask"] for x in tokenized_outputs]),
                "labels": torch.stack([x["labels"] for x in tokenized_outputs]),
            }

        # Apply processing to the dataset
        processed_dataset = dataset.map(
            process_function,
            batched=True,
            batch_size=8,  # Adjust batch size as needed
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )

        processed_dataset.set_format(type="torch")

        return processed_dataset

    def setup_peft_config(self):
        """
        Setup the PEFT configuration for the model.
        """
        # Find target modules by inspecting the model's named modules
        target_modules = self.configuration.get("target_modules", [])
        if target_modules == []:
            for name, module in self.model.named_modules():
                # Check if the module is of a supported type
                if isinstance(
                    module,
                    (
                        torch.nn.Linear,
                        torch.nn.Embedding,
                        torch.nn.Conv2d,
                        torch.nn.Conv3d,
                        transformers.pytorch_utils.Conv1D,
                    ),
                ):
                    target_modules.append(name)

        # Log the identified target modules
        logger.info(f"Identified target modules for LoRA: {target_modules}")

        return LoraConfig(
            r=self.configuration.get("r", 16),
            lora_alpha=self.configuration.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=self.configuration.get("lora_dropout", 0.05),
            bias="none",
            task_type=self.configuration.get("task_type", "CAUSAL_LM"),
        )

    def convert_from_dtlpy(self, data_path, **kwargs):
        """Convert Dataloop prompt items to training format."""
        dataset = self.model_entity.dataset
        
        # Download all items with annotations
        dataset.items.download(
            local_path=data_path, 
            annotation_options=dl.ViewAnnotationOptions.JSON,
            include_annotations_in_output=True
        )
        dataset.export(local_path=data_path, include_annotations=True)
        
        # Load exported metadata to get item info with tags
        entities = []
        for json_file in Path(data_path).glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entities.extend(data)
        
        # Split by train/validation tags
        train_messages = []
        validation_messages = []
        
        for entity in entities:
            mimetype = entity.get("metadata", {}).get("system", {}).get("mimetype", "")
            if mimetype != "application/json":
                continue
                
            subset_tags = entity.get("metadata", {}).get("system", {}).get("tags", {})
            entity_filepath = entity.get("filename", "").lstrip("/").split("/")
            prompt_path = os.path.join(data_path, "items", *entity_filepath)
            
            if not os.path.exists(prompt_path):
                continue
                
            # Read prompt item content
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
            
            # Extract messages from Dataloop prompt item structure
            messages = self._extract_messages_from_prompt_item(prompt_data, prompt_path)
            if not messages:
                continue
            
            if "train" in subset_tags:
                train_messages.append({"messages": messages})
            elif "validation" in subset_tags:
                validation_messages.append({"messages": messages})
        
        logger.info(f"Loaded {len(train_messages)} train and {len(validation_messages)} validation samples")
        
        if len(train_messages) == 0 or len(validation_messages) == 0:
            raise ValueError("Missing train or validation data. Check that items have train/validation tags.")
        
        # Create datasets
        from datasets import Dataset
        train_ds = Dataset.from_list(train_messages)
        eval_ds = Dataset.from_list(validation_messages)
        
        self.train_dataset = self.preprocess_chatbot_dataset(train_ds)
        self.eval_dataset = self.preprocess_chatbot_dataset(eval_ds)
    
    def _extract_messages_from_prompt_item(self, prompt_data, prompt_path):
        """Extract chat messages from Dataloop prompt item structure."""
        messages = []
        
        # Check if it's already in simple messages format
        if "messages" in prompt_data and isinstance(prompt_data["messages"], list):
            return prompt_data["messages"]
        
        # Handle Dataloop prompt item format with prompts dict
        prompts = prompt_data.get("prompts", {})
        if isinstance(prompts, dict):
            for prompt_key, prompt_content in prompts.items():
                # Extract user message from prompt elements
                elements = prompt_content if isinstance(prompt_content, list) else []
                for element in elements:
                    if element.get("mimetype") in ["text", "application/text", dl.PromptType.TEXT]:
                        messages.append({
                            "role": "user",
                            "content": element.get("value", "")
                        })
        
        # Read annotations file for assistant response (take only first text annotation)
        annotation_path = prompt_path.replace("items", "json")
        if os.path.exists(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotations_data = json.load(f)
            
            for annotation in annotations_data.get("annotations", []):
                if annotation.get("type") == "text":
                    messages.append({
                        "role": "assistant",
                        "content": annotation.get("coordinates", "")
                    })
                    break  # Only take first annotation
        
        return messages

    def train(self, data_path, output_path, **kwargs):
        # Make sure CUDA is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available! Check your GPU configuration.")
        
        # Prepare model and tokenizer for finetune
        self.logger.info(f"Preparing model and tokenizer for finetune from {self.model_path}")
        self.model = self.prepare_model_for_finetune(model_path=self.model_path)

        # Start GPU monitoring in a separate thread
        torch.cuda.empty_cache()
        monitor_thread = threading.Thread(target=self.keep, daemon=True)
        monitor_thread.start()

        peft_config = self.setup_peft_config()
        self.peft_model = get_peft_model(self.model, peft_config)
        logger.info(self.peft_model.print_trainable_parameters())

        # Training arguments setup
        num_train_epochs = self.configuration.get("num_train_epochs", 1)
        per_device_train_batch_size = self.configuration.get("per_device_train_batch_size", 1)
        gradient_accumulation_steps = self.configuration.get("gradient_accumulation_steps", 16)
        optim = self.configuration.get("optim", "paged_adamw_32bit")
        save_steps = self.configuration.get("save_steps", 10)
        logging_steps = self.configuration.get("logging_steps", 10)
        learning_rate = self.configuration.get("learning_rate", 2e-4)
        warmup_ratio = self.configuration.get("warmup_ratio", 0.03)
        lr_scheduler_type = self.configuration.get("lr_scheduler_type", "constant")
        bf16 = self.configuration.get("bf16", True)
        group_by_length = self.configuration.get("group_by_length", True)
        save_total_limit = self.configuration.get("save_total_limit", 3)
        max_grad_norm = self.configuration.get("max_grad_norm", 0.3)
        remove_unused_columns = self.configuration.get("remove_unused_columns", False)
        gradient_checkpointing = self.configuration.get("gradient_checkpointing", True)
        use_reentrant = self.configuration.get("use_reentrant", False)
        report_to = self.configuration.get("report_to", ["tensorboard"])
        logging_first_step = self.configuration.get("logging_first_step", True)
        log_level = self.configuration.get("log_level", "info")
        logging_strategy = self.configuration.get("logging_strategy", "steps")

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            bf16=bf16,
            group_by_length=group_by_length,
            save_total_limit=save_total_limit,
            max_grad_norm=max_grad_norm,
            remove_unused_columns=remove_unused_columns,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to=report_to,
            logging_dir=os.path.join(output_path, "logs"),
            logging_first_step=logging_first_step,
            log_level=log_level,
            logging_strategy=logging_strategy,
        )

        logger.info(f"\nTraining arguments:\n{training_args}")

        try:
            # Create callback for epoch-end saving
            save_callback = SaveEpochCallback(
                save_path=os.path.join(output_path, "checkpoints"),
                tokenizer=self.tokenizer,  # Pass the tokenizer
                save_every_n_epochs=self.configuration.get("save_every_n_epochs", 1),
            )

            # Create trainer and start training
            self.trainer = Trainer(
                model=self.peft_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=training_args,
                callbacks=[save_callback],
            )
            self.trainer.train()

        except Exception as e:
            raise Exception(f"Training error: {e}") from e

        finally:
            # Training finished or error - stop monitoring
            self.stop_monitoring = True
            monitor_thread.join(timeout=5)  # Wait for monitoring thread to finish

    def save(self, local_path, **kwargs):
        """
        Saves the fine-tuned model and tokenizer to the specified path.

        Args:
            local_path (str): Path where the model will be saved.
            **kwargs: Additional arguments.
        """
        final_save_path = os.path.join(local_path, "best")
        
        # Save peft_model (LoRA adapters) after training
        if hasattr(self, 'peft_model') and self.peft_model is not None:
            self.peft_model.save_pretrained(save_directory=final_save_path)
            self.logger.info("Saved fine-tuned PEFT model.")
        else:
            raise ValueError("No model found to save. Please ensure model is trained first.")
            
        self.tokenizer.save_pretrained(save_directory=final_save_path)
        # Extract model name without namespace to avoid path duplication
        model_name_only = self.model_entity.name.split('/')[-1] if '/' in self.model_entity.name else self.model_entity.name
        self.configuration["model_path"] = f"{model_name_only}/best"

    def get_gpu_memory(self):
        """
        Retrieves current GPU memory usage information.

        Returns:
            tuple: Three lists containing free, total, and used memory for each GPU.
        """
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        info = subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        free = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        info = subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        total = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        info = subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        used = [int(x.split()[0]) for i, x in enumerate(info)]
        return free, total, used

    def keep(self):
        """
        Monitors GPU memory usage during training in a separate thread.

        Continuously logs GPU memory statistics until training completes.
        """
        self.stop_monitoring = False
        while not self.stop_monitoring:
            try:
                free, total, used = self.get_gpu_memory()
                logger.info(f"GPU Memory - Total: {total}MB, Used: {used}MB, Free: {free}MB")
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error monitoring GPU: {e}")
                break

    @staticmethod
    def reformat_messages(messages):
        """
        Convert OpenAI message format to HTTP request format.
        This function takes messages formatted for OpenAI's API and transforms them into a format suitable for HTTP
        requests.

        :param messages: A list of messages in the OpenAI format.
        :return: A list of messages reformatted for HTTP requests.
        """
        reformat_messages = list()
        for message in messages:
            content = message["content"]
            question = content[0][content[0].get("type")]
            role = message["role"]

            reformat_message = {"role": role, "content": question}
            reformat_messages.append(reformat_message)
        return reformat_messages

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch, **kwargs):
        """
        Performs inference on a batch of items using the loaded model.

        Processes each prompt item in the batch, generates responses using
        the model, and adds the responses back to the prompt items.

        Args:
            batch: List of prompt items to process.
            **kwargs: Additional arguments.

        Returns:
            list: Empty list (annotations are added directly to prompt items).
        """
        system_prompt = self.model_entity.configuration.get("system_prompt", "")
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name
        # Retrieve generation parameters from configuration
        max_new_tokens = self.configuration.get("max_new_tokens", 512)
        temperature = self.configuration.get("temperature", 0.7)
        do_sample = self.configuration.get("do_sample", True)
        top_p = self.configuration.get("top_p", 0.95)
        repetition_penalty = self.configuration.get("repetition_penalty", 1.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predicting on device: {device}")
        
        for prompt_item in batch:
            # Get all messages including model annotations
            _messages = prompt_item.to_messages(model_name=model_name)
            messages = self.reformat_messages(_messages)
            messages.insert(0, {"role": "system", "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get("nearestItems", [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items, add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            # Format the input using the chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate response - ensure inputs are on same device as model
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Get device from model (handles device_map="auto" case)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Decode correctly, removing input text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            response_text = response[len(input_text):].strip()
            
            prompt_item.add(
                message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response_text}]},
                model_info={"name": self.model_entity.name, "confidence": 1.0, "model_id": self.model_entity.id},
            )

        return []


class SaveEpochCallback(TrainerCallback):
    """Custom callback to save model after each epoch."""

    def __init__(self, save_path, tokenizer, save_every_n_epochs=1):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Callback triggered at the end of each training epoch.

        Saves the model and tokenizer periodically based on the specified frequency.

        Args:
            args: Training arguments.
            state: The current training state.
            control: TrainerControl object.
            **kwargs: Additional arguments including the model.
        """
        epoch = int(state.epoch)

        if epoch % self.save_every_n_epochs == 0:
            logger.info(f"\nSaving model and tokenizer for epoch {epoch}")

            # Get model from kwargs
            model = kwargs.get("model", None)

            if model is None:
                logger.error("Cannot save model. Model is None.")
                return

            # Create epoch-specific directory
            epoch_dir = os.path.join(self.save_path, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(save_directory=epoch_dir)
            self.tokenizer.save_pretrained(save_directory=epoch_dir)

            logger.info(f"Model and tokenizer saved in {epoch_dir}")
        else:
            logger.info(f"Skipping save for epoch {epoch} (Only saving every {self.save_every_n_epochs} epochs)")
