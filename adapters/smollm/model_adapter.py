import dtlpy as dl
import torch
import logging
import os
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

logger = logging.getLogger("[smollm-1-7b]")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        logger.info("Loading SmolLM adapter...")
        self.model_path = self.configuration.get("model_path", "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.max_tokens = self.configuration.get('max_tokens', 1024)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.9)
        self.stream = self.configuration.get('stream', True)
        self.system_prompt = self.configuration.get('system_prompt', None)
        
        logger.info(f"Model configuration - Path: {self.model_path}, Max tokens: {self.max_tokens}, Temperature: {self.temperature}, Top-p: {self.top_p}, Stream: {self.stream}")
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info(f"Loading tokenizer from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            
            # Prepare model (with optional LoRA)
            base_model = self._prepare_model(ckpt=self.model_path)
            
            # Check if LoRA adapter weights exist
            lora_adapter_path = os.path.join(local_path, "adapter_config.json")
            if os.path.exists(lora_adapter_path):
                logger.info(f"Loading LoRA weights from {local_path}")
                model_with_lora = PeftModel.from_pretrained(base_model, local_path)
                self.model = model_with_lora.merge_and_unload()
                logger.info("LoRA weights loaded and merged successfully")
            else:
                self.model = base_model
                logger.info("No LoRA weights found, using base model")
                
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def save(self, local_path, **kwargs):
        """Save the model and tokenizer to Dataloop"""
        self.model.save_pretrained(save_directory=local_path)
        self.tokenizer.save_pretrained(save_directory=local_path)
        logger.info(f"Successfully saved model and tokenizer to {local_path}")

    def prepare_item_func(self, item: dl.Item):
        logger.debug(f"Preparing item: {item.id}")
        prompt_item = dl.PromptItem.from_item(item=item)
        logger.debug(f"Created prompt item with {len(prompt_item.prompts)} prompts")
        return prompt_item

    def train(self, data_path, output_path, **kwargs):
        """Train the model using the provided training data."""
        logger.info("Starting SmolLM training")
        
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        try:
            configs = self.model_entity.configuration
            
            # Configure training arguments
            num_train_epochs = configs.get("num_train_epochs", 3)
            per_device_train_batch_size = configs.get("per_device_train_batch_size", 4)
            gradient_accumulation_steps = configs.get("gradient_accumulation_steps", 4)
            warmup_steps = configs.get("warmup_steps", 100)
            learning_rate = configs.get("learning_rate", 2e-4)
            weight_decay = configs.get("weight_decay", 0.01)
            adam_beta2 = configs.get("adam_beta2", 0.999)
            optim = configs.get("optim", "adamw_torch")
            bf16 = configs.get("bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            fp16 = configs.get("fp16", torch.cuda.is_available() and not bf16)
            save_strategy = configs.get("save_strategy", "epoch")
            logging_strategy = configs.get("logging_strategy", "steps")
            logging_steps = configs.get("logging_steps", 10)
            eval_strategy = configs.get("evaluation_strategy", "epoch")
            save_total_limit = configs.get("save_total_limit", 2)
            gradient_checkpointing = configs.get("gradient_checkpointing", True)
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                           f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            
            # Configure training arguments with memory optimizations
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                adam_beta2=adam_beta2,
                optim=optim,
                bf16=bf16,
                fp16=fp16,
                save_strategy=save_strategy,
                logging_strategy=logging_strategy,
                logging_steps=logging_steps,
                eval_strategy=eval_strategy,
                save_total_limit=save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                push_to_hub=False,
                report_to=["none"],
                gradient_checkpointing=gradient_checkpointing,
                # Memory optimization settings
                dataloader_pin_memory=False,  # Can help with memory fragmentation
                remove_unused_columns=False,  # Keep all columns
            )
            
            # Create save callback
            save_callback = SaveEpochCallback(
                save_path=output_path, 
                tokenizer=self.tokenizer, 
                model_entity=self.model_entity
            )
            
            # Get training data
            logger.info("Loading and processing training data...")
            train_dataset, val_dataset = self.convert_from_dtlpy(data_path)
            logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            # Initialize trainer
            # Ensure all model parameters are trainable before initializing Trainer
            params_reenabled = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    param.requires_grad_(True)
                    params_reenabled += 1
            if params_reenabled:
                logger.info(f"Re-enabled gradients for {params_reenabled} parameters")

            trainer = Trainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=training_args,
                data_collator=self._data_collator,
                callbacks=[save_callback],
            )
            
            # Train
            logger.info("Starting training...")
            
            # Memory usage callback
            def log_gpu_memory():
                if torch.cuda.is_available():
                    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                               f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            
            # Log memory periodically during training
            if kwargs.get("on_epoch_end_callback"):
                original_callback = kwargs["on_epoch_end_callback"]
                def combined_callback(i_epoch, n_epoch):
                    log_gpu_memory()
                    original_callback(i_epoch, n_epoch)
                kwargs["on_epoch_end_callback"] = combined_callback
            
            trainer.train()
            
            # Save the final model
            logger.info("Saving final model...")
            trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info("Training completed successfully")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU Out of Memory Error: {str(e)}")
            logger.error("Try reducing batch_size or gradient_accumulation_steps")
            raise
            
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
            
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared after training")

    def _reformat_messages(self, messages):
        """Convert messages to format expected by the model."""
        logger.debug(f"Reformatting {len(messages)} messages")
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg.get('role', 'user')
            content = msg.get('content', [])
            
            # Extract text content from the message
            text_content = ""
            if isinstance(content, str):
                text_content = content
                logger.debug(f"Message {i}: {role} - string content ({len(text_content)} chars)")
            elif isinstance(content, list):
                for element in content:
                    if isinstance(element, dict) and element.get('type') == 'text':
                        text_content += element.get('text', '')
                    elif isinstance(element, dict) and element.get('mimetype') == dl.PromptType.TEXT:
                        text_content += element.get('value', '')
                logger.debug(f"Message {i}: {role} - list content ({len(text_content)} chars)")
            
            if text_content:
                formatted_messages.append({
                    'role': role,
                    'content': text_content
                })
            else:
                logger.warning(f"Message {i} has no text content")
        
        logger.debug(f"Reformatted to {len(formatted_messages)} messages")
        return formatted_messages
    
    def _format_chat_ml(self, messages):
        """Manual ChatML formatting as alternative to apply_chat_template."""
        formatted = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add generation prompt
        formatted += "<|im_start|>assistant\n"
        
        logger.debug(f"Manual ChatML formatting complete: {len(formatted)} chars")
        return formatted
    
    def _get_response(self, messages):
        """Generate response from the model."""
        logger.debug("Generating response from model")
        try:
            # Manual ChatML formatting as alternative to apply_chat_template
            input_text = self._format_chat_ml(messages)
            logger.debug(f"Applied manual chat formatting, input length: {len(input_text)} chars")
            
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            logger.debug(f"Tokenized input, tensor shape: {inputs.shape}")
            
            logger.info(f"Starting generation with max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            output_text = self.tokenizer.decode(outputs[0])
            logger.debug(f"Generated output length: {len(output_text)} chars")
            
            lines = output_text.split('<|im_end|>')
            last_assistant_response = None
            
            # Extract the last assistant response
            for line in lines:
                if '<|im_start|>assistant' in line:
                    last_assistant_response = line.split('<|im_start|>assistant\n')[-1].strip()
            
            if last_assistant_response:
                logger.info(f"Extracted assistant response ({len(last_assistant_response)} chars)")
            else:
                logger.warning("Could not extract assistant response from generated text")
            
            return last_assistant_response or ""
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise
    
    def _stream_response(self, messages):
        """Generate streaming response from the model."""
        logger.debug("Starting streaming response")
        response = self._get_response(messages)
        
        # Simulate streaming by yielding chunks
        chunk_size = 10  # Characters per chunk
        total_chunks = (len(response) + chunk_size - 1) // chunk_size
        logger.debug(f"Streaming response in {total_chunks} chunks of {chunk_size} chars each")
        
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            logger.debug(f"Yielding chunk {i//chunk_size + 1}/{total_chunks}")
            yield chunk
    
    def predict(self, batch, **kwargs):
        logger.info(f"Starting prediction for batch of {len(batch)} items")
        
        # Validate batch
        if not batch:
            logger.warning("Empty batch received, returning empty list")
            return []
        
        # Check if model and tokenizer are loaded
        if not hasattr(self, 'model') or self.model is None:
            logger.error("Model not loaded. Please call load() first.")
            raise RuntimeError("Model not loaded. Cannot perform prediction.")
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("Tokenizer not loaded. Please call load() first.")
            raise RuntimeError("Tokenizer not loaded. Cannot perform prediction.")
        
        annotations = []
        
        for idx, prompt_item in enumerate(batch):
            try:
                logger.info(f"Processing item {idx + 1}/{len(batch)}")
                
                # Validate prompt_item
                if not hasattr(prompt_item, 'to_messages'):
                    logger.error(f"Item {idx} is not a valid PromptItem")
                    continue
                
                prompt_item: dl.PromptItem = prompt_item
                
                # Get messages with error handling
                try:
                    _messages = prompt_item.to_messages(model_name=self.model_entity.name)
                    logger.debug(f"Converted to {len(_messages)} messages")
                except Exception as e:
                    logger.error(f"Failed to convert prompt item to messages: {str(e)}")
                    continue
                
                # Reformat messages
                try:
                    messages = self._reformat_messages(_messages)
                except Exception as e:
                    logger.error(f"Failed to reformat messages: {str(e)}")
                    continue
                
                # Add system prompt if configured
                if self.system_prompt is not None:
                    logger.debug("Adding system prompt to messages")
                    messages.insert(0, {"role": "system", "content": self.system_prompt})
                
                # Handle nearest items context
                try:
                    nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', []) if prompt_item.prompts else []
                    if len(nearest_items) > 0:
                        logger.info(f"Found {len(nearest_items)} nearest items, building context")
                        context = prompt_item.build_context(nearest_items=nearest_items,
                                                            add_metadata=self.configuration.get("add_metadata"))
                        logger.info(f"Nearest items Context: {context}")
                        messages.append({"role": "assistant", "content": context})
                except Exception as e:
                    logger.warning(f"Failed to build context from nearest items: {str(e)}")
                    # Continue without context

                # Generate response
                response = ""
                try:
                    if self.stream:
                        logger.info("Using streaming mode")
                        stream_response = self._stream_response(messages=messages)
                        chunk_count = 0
                        for chunk in stream_response:
                            response += chunk
                            chunk_count += 1
                        logger.info(f"Received {chunk_count} chunks, total response length: {len(response)} chars")
                    else:
                        logger.info("Using non-streaming mode")
                        response = self._get_response(messages=messages)
                        logger.info(f"Generated response length: {len(response)} chars")
                    
                    # Add response to prompt item
                    if response:
                        prompt_item.add(message={"role": "assistant",
                                                 "content": [{"mimetype": dl.PromptType.TEXT,
                                                              "value": response}]},
                                        model_info={'name': self.model_entity.name,
                                                    'confidence': 1.0,
                                                    'model_id': self.model_entity.id})
                        logger.debug(f"Successfully added response to item {idx + 1}")
                    else:
                        logger.warning(f"Empty response generated for item {idx + 1}")
                        
                except Exception as e:
                    logger.error(f"Failed to generate response for item {idx + 1}: {str(e)}")
                    # Add error response
                    error_message = f"Error generating response: {str(e)}"
                    prompt_item.add(message={"role": "assistant",
                                             "content": [{"mimetype": dl.PromptType.TEXT,
                                                          "value": error_message}]},
                                    model_info={'name': self.model_entity.name,
                                                'confidence': 0.0,
                                                'model_id': self.model_entity.id})
                
            except Exception as e:
                logger.error(f"Unexpected error processing item {idx + 1}: {str(e)}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        
        logger.info("Prediction batch completed")
        return annotations
    
    def convert_from_dtlpy(self, data_path, **kwargs):
        """Convert Dataloop data to format suitable for training."""
        logger.info(f"Converting data from Dataloop format at: {data_path}")
        
        try:
            # Validate data_path exists
            if not os.path.exists(data_path):
                raise ValueError(f"Data path does not exist: {data_path}")
            
            # Subsets validation
            subsets = self.model_entity.metadata.get("system", {}).get("subsets", None)
            if subsets is None:
                raise ValueError(
                    "No subsets defined in model metadata. "
                    "Please define 'train' and 'validation' subsets in the model's system metadata."
                )
            
            if not isinstance(subsets, dict):
                raise ValueError(
                    f"Expected subsets to be a dict, got {type(subsets)}. "
                    "Please check the model's system metadata."
                )
            
            # Check required subsets
            missing_subsets = []
            if "train" not in subsets:
                missing_subsets.append("train")
            if "validation" not in subsets:
                missing_subsets.append("validation")
            
            if missing_subsets:
                raise ValueError(
                    f"Missing required subsets: {', '.join(missing_subsets)}. "
                    f"SmolLM requires both 'train' and 'validation' subsets. "
                    f"Current subsets: {list(subsets.keys())}"
                )
            
            train_texts = []
            train_labels = []
            val_texts = []
            val_labels = []
            
            # Process each subset
            for subset, filters_dict in subsets.items():
                if subset not in ["train", "validation"]:
                    logger.warning(f"Skipping unknown subset: {subset}")
                    continue
                
                logger.info(f"Processing subset: {subset}")
                data_subset_base_path = os.path.join(data_path, subset)
                
                # Validate filters_dict
                if not isinstance(filters_dict, dict):
                    logger.error(f"Invalid filter for subset {subset}: expected dict, got {type(filters_dict)}")
                    raise ValueError(f"Invalid filter configuration for subset {subset}")
                
                # Add json type validation
                filters_dict = filters_dict.copy()  # Don't modify original
                
                # Ensure filter structure exists
                if "filter" not in filters_dict:
                    filters_dict["filter"] = {"$and": []}
                elif "$and" not in filters_dict.get("filter", {}):
                    filters_dict["filter"] = {"$and": []}
                
                # Add JSON mimetype condition
                new_condition = {"metadata.system.mimetype": {"$eq": "application/json"}}
                if new_condition not in filters_dict["filter"]["$and"]:
                    filters_dict["filter"]["$and"].append(new_condition)
                
                # Create filters
                try:
                    filters = dl.Filters(custom_filter=filters_dict)
                except Exception as e:
                    logger.error(f"Failed to create filters for subset {subset}: {str(e)}")
                    raise ValueError(f"Invalid filter configuration for subset {subset}: {str(e)}")
                
                # List items
                try:
                    pages = self.model_entity.dataset.items.list(filters=filters)
                    items_count = pages.items_count
                except Exception as e:
                    logger.error(f"Failed to list items for subset {subset}: {str(e)}")
                    raise RuntimeError(f"Failed to list items for subset {subset}: {str(e)}")
                
                if items_count == 0:
                    raise ValueError(
                        f"Found 0 json files in subset '{subset}'. "
                        f"SmolLM training expects prompt items in JSON format. "
                        f"Please ensure your dataset contains JSON prompt items matching the filter."
                    )
                
                logger.info(f"Found {items_count} items in {subset} subset")
                
                # Download items
                try:
                    logger.info(f"Downloading items to: {data_subset_base_path}")
                    download_result = self.model_entity.dataset.items.download(
                        filters=filters, 
                        local_path=data_subset_base_path,
                        to_items_folder=True,
                        overwrite=True
                    )
                    logger.info(f"Download completed for subset {subset}")
                except Exception as e:
                    logger.error(f"Failed to download items for subset {subset}: {str(e)}")
                    raise RuntimeError(f"Failed to download items for subset {subset}: {str(e)}")
                
                # Process downloaded items
                try:
                    texts, labels = self._process_prompt_items(data_subset_base_path)
                    
                    if not texts:
                        raise ValueError(
                            f"No valid training data found in subset '{subset}'. "
                            f"All downloaded files were invalid or empty."
                        )
                    
                    logger.info(f"Successfully processed {len(texts)} examples from subset {subset}")
                    
                    if subset == "train":
                        train_texts = texts
                        train_labels = labels
                    elif subset == "validation":
                        val_texts = texts
                        val_labels = labels
                        
                except Exception as e:
                    logger.error(f"Failed to process items for subset {subset}: {str(e)}")
                    raise
            
            # Final validation
            if not train_texts:
                raise ValueError("No training examples found after processing all subsets")
            if not val_texts:
                raise ValueError("No validation examples found after processing all subsets")
            
            logger.info(f"Final dataset sizes - Training: {len(train_texts)}, Validation: {len(val_texts)}")
            
            # Convert to Dataset objects
            try:
                train_dataset = Dataset.from_dict({
                    "text": train_texts,
                    "labels": train_labels
                })
                
                val_dataset = Dataset.from_dict({
                    "text": val_texts,
                    "labels": val_labels
                })
                
                logger.info("Successfully created HuggingFace datasets")
                return train_dataset, val_dataset
                
            except Exception as e:
                logger.error(f"Failed to create HuggingFace datasets: {str(e)}")
                raise RuntimeError(f"Failed to create HuggingFace datasets: {str(e)}")
                
        except Exception as e:
            logger.error(f"convert_from_dtlpy failed: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _process_prompt_items(self, data_path):
        """Process downloaded prompt items to extract conversations."""
        texts = []
        labels = []
        
        items_path = Path(data_path) / "items"
        
        # Check if items directory exists
        if not items_path.exists():
            logger.error(f"Items directory not found: {items_path}")
            raise ValueError(f"Items directory not found: {items_path}")
        
        json_files = list(items_path.rglob("*.json"))
        
        if not json_files:
            logger.error(f"No JSON files found in {items_path}")
            raise ValueError(f"No JSON files found in {items_path}")
        
        logger.info(f"Processing {len(json_files)} JSON files from {items_path}")
        
        valid_files = 0
        skipped_files = 0
        
        for json_file in json_files:
            try:
                # Read and validate JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {json_file}: {str(e)}")
                        skipped_files += 1
                        continue
                
                # Validate structure
                if not isinstance(data, dict):
                    logger.warning(f"Expected dict, got {type(data)} in {json_file}")
                    skipped_files += 1
                    continue
                
                # Check for prompts field
                if "prompts" not in data:
                    logger.warning(f"No 'prompts' field found in {json_file}")
                    skipped_files += 1
                    continue
                
                prompts = data["prompts"]
                if not isinstance(prompts, dict):
                    logger.warning(f"'prompts' field is not a dict in {json_file}")
                    skipped_files += 1
                    continue
                
                messages = []
                
                # Process each prompt to build conversation
                for prompt_key, prompt_data in prompts.items():
                    if not isinstance(prompt_data, list):
                        logger.warning(f"Prompt data for key '{prompt_key}' is not a list in {json_file}")
                        continue
                    
                    for element in prompt_data:
                        if not isinstance(element, dict):
                            logger.warning(f"Element is not a dict in prompt '{prompt_key}' in {json_file}")
                            continue
                        
                        mimetype = element.get("mimetype", "")
                        
                        # Handle different mimetypes
                        if mimetype == "application/text" or mimetype == dl.PromptType.TEXT:
                            value = element.get("value", "")
                            if not value or not isinstance(value, str):
                                logger.debug(f"Skipping empty or non-string text value in {json_file}")
                                continue
                            
                            # Extract role from metadata if available
                            role = "user"  # Default role
                            if mimetype == "metadata" and isinstance(element.get("value"), dict):
                                role = element["value"].get("role", "user")
                            
                            messages.append({"role": role, "content": value})
                        
                        elif mimetype == "metadata":
                            # Extract role from metadata
                            metadata = element.get("value", {})
                            if isinstance(metadata, dict) and "role" in metadata:
                                # Update the role for the last message if available
                                if messages and "role" in metadata:
                                    messages[-1]["role"] = metadata["role"]
                
                if messages:
                    # Apply chat template to create training text
                    try:
                        # Use manual formatting as fallback
                        try:
                            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                        except Exception as template_error:
                            logger.debug(f"Chat template failed, using manual formatting: {str(template_error)}")
                            text = self._format_chat_ml(messages)
                        
                        texts.append(text)
                        labels.append(text)  # For language modeling, labels are same as input
                        valid_files += 1
                        logger.debug(f"Successfully processed {json_file} with {len(messages)} messages")
                    except Exception as e:
                        logger.warning(f"Failed to process chat messages for {json_file}: {str(e)}")
                        skipped_files += 1
                else:
                    logger.warning(f"No valid messages found in {json_file}")
                    skipped_files += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {str(e)}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                skipped_files += 1
                continue
        
        logger.info(f"Processing complete: {valid_files} valid files, {skipped_files} skipped files")
        
        if not texts:
            raise ValueError(f"No valid training data found in {data_path}. All {len(json_files)} files were invalid or empty.")
        
        return texts, labels
    
    def _data_collator(self, examples):
        """Prepare batch for training."""
        texts = [example["text"] for example in examples]
        
        # Tokenize the texts
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.configuration.get("max_length", 2048),
            return_tensors="pt"
        )
        
        # For language modeling, labels are the input_ids shifted by one position
        batch["labels"] = batch["input_ids"].clone()
        
        # Replace padding token id with -100 so it's ignored in loss computation
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch
    
    def _prepare_model(self, ckpt):
        """Prepare model with optional LoRA configuration."""
        use_lora = self.configuration.get("use_lora", True)
        
        logger.info(f"Loading base model from {ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if use_lora:
            logger.info("Configuring LoRA")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.configuration.get("lora_r", 16),
                lora_alpha=self.configuration.get("lora_alpha", 32),
                lora_dropout=self.configuration.get("lora_dropout", 0.1),
                target_modules=self.configuration.get(
                    "target_modules", 
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                ),
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        if self.device != "cuda":
            model = model.to(self.device)
            
        return model


class SaveEpochCallback(TrainerCallback):
    """Custom callback to save model after each epoch and track eval loss."""
    
    def __init__(self, save_path, tokenizer, model_entity, save_every_n_epochs=1):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.model_entity = model_entity
        self.save_every_n_epochs = save_every_n_epochs
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.current_loss = None
        self.current_epoch = 0.0  # Initialize as float to prevent None issues
        self.eval_losses = []
        self.log_file = os.path.join(self.save_path, "training_logs.json")
        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.current_loss = logs["loss"]
            self.current_epoch = state.epoch if state.epoch is not None else 0
            
            if self.best_loss == float("inf"):
                self.best_loss = self.current_loss
                logger.info(f"Initial best loss set to: {self.best_loss:.4f}")
            
            # Log metrics to Dataloop
            try:
                self.model_entity.metrics.create(
                    samples=dl.PlotSample(
                        figure="loss", 
                        legend="train", 
                        x=self.current_epoch, 
                        y=self.current_loss
                    ),
                    dataset_id=self.model_entity.dataset_id,
                )
            except Exception as e:
                logger.warning(f"Failed to log metrics to Dataloop: {str(e)}")
            
            # Save training logs
            current_logs = state.log_history
            with open(self.log_file, "w") as f:
                json.dump(current_logs, f, indent=2)
            logger.debug(f"Updated training logs in {self.log_file}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            self.eval_losses.append(eval_loss)
            
            logger.info(f"Epoch {self.current_epoch:.2f} - Eval Loss: {eval_loss:.4f}")
            
            # Log eval metrics to Dataloop
            try:
                self.model_entity.metrics.create(
                    samples=dl.PlotSample(
                        figure="loss", 
                        legend="validation", 
                        x=self.current_epoch, 
                        y=eval_loss
                    ),
                    dataset_id=self.model_entity.dataset_id,
                )
            except Exception as e:
                logger.warning(f"Failed to log eval metrics to Dataloop: {str(e)}")
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            logger.warning("Model is None in on_epoch_end callback")
            return
        
        # Get current epoch - handle potential None
        current_epoch = int(state.epoch) if state.epoch is not None else self.current_epoch
        self.current_epoch = current_epoch
        
        if current_epoch > 0 and current_epoch % self.save_every_n_epochs == 0:
            logger.info(f"Epoch {current_epoch} completed. Current loss: {self.current_loss}")
            
            # Always save checkpoint at specified intervals
            checkpoint_dir = os.path.join(self.save_path, f"checkpoint-epoch-{current_epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Checkpoint saved in {checkpoint_dir}")
            
            # Save best model if this is the best so far
            if self.current_loss is not None and self.current_loss < self.best_loss:
                logger.info(f"New best model found! Loss: {self.current_loss:.4f} (previous: {self.best_loss:.4f})")
                self.best_loss = self.current_loss
                self.best_epoch = current_epoch
                
                # Save best model
                best_model_dir = os.path.join(self.save_path, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                logger.info(f"Best model saved in {best_model_dir}")


if __name__ == "__main__":
    import dtlpy as dl
    dl.setenv('prod')
    if dl.token_expired():
        dl.login()
    project: dl.Project = dl.projects.get(project_id='<project_id>')
    dataset: dl.Dataset = dl.datasets.get(dataset_id='<dataset_id>')
    model_entity = project.models.get(model_id="<model_id>")
    item = dl.items.get(item_id="<item_id>")
    adapter = ModelAdapter(model_entity=model_entity)
    prompt_item = dl.PromptItem.from_item(item=item)
    response = adapter.predict([prompt_item])
    print(response)
