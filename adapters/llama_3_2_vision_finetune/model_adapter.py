import os
import json
import time
import torch
import base64
import logging
import dtlpy as dl
import gc

from PIL import Image
from io import BytesIO
from pathlib import Path
from datasets import Dataset
from datetime import datetime
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoProcessor, MllamaForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments

logger = logging.getLogger("llama-vision-finetune")


class ModelAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for Llama Vision model from Meta
    """

    def load(self, local_path, **kwargs):
        """Load the model and processor from local path."""
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if self.hf_token is None:
            raise ValueError("Missing HUGGINGFACE_TOKEN environment variable")
        login(token=self.hf_token)

        hf_model_name = self.model_entity.configuration.get("model_name", "meta-llama/Llama-3.2-11B-Vision")
        logger.info(f"Model name: {hf_model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Using device: {self.device}')
        self.device_map = self.configuration.get("device_map", "auto")
        self.prompt_items_dir = self.configuration.get("prompt_items_dir", "/prompt_items")

        # load base model
        logger.info(f"Downloading model from HuggingFace {hf_model_name}")
        base_model, self.processor = self._prepare_model_and_tokenizer(ckpt=hf_model_name)

        # check if lora adapter weights exist
        if os.path.exists(local_path) is True:
            logger.info(f"Loading LoRA weights from {local_path}")
            model_to_merge = PeftModel.from_pretrained(base_model, local_path, device_map=self.device_map)
            merged_model = model_to_merge.merge_and_unload()
            self.model = merged_model
        else:
            self.model = base_model

    def save(self, local_path, **kwargs):
        """Save the model and processor to Dataloop"""
        self.model.save_pretrained(save_directory=local_path)
        self.processor.save_pretrained(save_directory=local_path)
        logger.info(f"Successfully saved trained LoRA weights to {local_path}")

    def prepare_item_func(self, item: dl.Item):
        """Prepare item for prediction"""
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch, **kwargs):
        """Make predictions using the model."""
        try:
            system_prompt = self.model_entity.configuration.get("system_prompt", None)
            # Retrieve generation parameters from configuration
            max_new_tokens = self.configuration.get("max_new_tokens", 512)
            temperature = self.configuration.get("temperature", 0.7)
            do_sample = self.configuration.get("do_sample", True)
            use_cache = self.configuration.get("use_cache", True)
            top_p = self.configuration.get("top_p", 0.9)
            # repetition_penalty = self.configuration.get("repetition_penalty", 1.1) # https://discuss.huggingface.co/t/issues-when-fine-tuning-llama-3-2-11b-vision/153972

            logger.info(f"Predicting on device: {self.device}")
            if self.device == "cuda":
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            with torch.amp.autocast(device_type=self.device.type):
                for prompt_item in batch:
                    try:
                        _messages = prompt_item.to_messages()
                        prompt_txt, image = self._reformat_messages(_messages, system_prompt)

                        logger.info(f"Prompt text: {prompt_txt}")
                        logger.info(f"Processing image of size: {image.size}")

                        # prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Answer briefly. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{prompt_txt}<|eot_id|>"
                        prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_txt}]
                        prompt = self.processor.apply_chat_template(
                            [{"role": "user", "content": prompt_content}], add_generation_prompt=True
                        )

                        # Process the image and text together
                        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(
                            self.device
                        )
                        logger.info("Generating response...")
                        # start_time = time.time()
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=do_sample,
                            use_cache=use_cache,
                            top_p=top_p,
                        ).to(self.device)
                        # end_time = time.time()
                        # logger.info(f"Time taken: {end_time - start_time} seconds")
                        # response = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
                        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])

                        logger.info(f"Response: {response}")
                        prompt_item.add(
                            message={
                                "role": "assistant",
                                "content": [{"mimetype": dl.PromptType.TEXT, "value": response}],
                            },
                            model_info={
                                "name": self.model_entity.name,
                                "confidence": 1.0,
                                "model_id": self.model_entity.id,
                            },
                        )

                        # Clear memory after processing each item
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            del outputs
                            del inputs
                            torch.cuda.synchronize()
                            gc.collect()  # TODO test this by loading images and deleting
                    except Exception as e:
                        logger.error(f"Error processing item {prompt_item.id}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

        return []

    def train(self, data_path, output_path, **kwargs):
        """Train the model using the provided training data."""
        configs = self.model_entity.configuration
        # Configure training arguments
        num_train_epochs = configs.get("num_train_epochs", 10)
        save_every_n_epochs = configs.get("save_every_n_epochs", 1)
        per_device_train_batch_size = configs.get("per_device_train_batch_size", 5)
        gradient_accumulation_steps = configs.get("gradient_accumulation_steps", 16)
        warmup_steps = configs.get("warmup_steps", 2)
        learning_rate = configs.get("learning_rate", 2e-4)
        weight_decay = configs.get("weight_decay", 1e-6)
        adam_beta2 = configs.get("adam_beta2", 0.999)
        optim = configs.get("optim", "paged_adamw_32bit")
        bf16 = configs.get("bf16", True)
        save_strategy = configs.get("save_strategy", "epoch")
        logging_strategy = configs.get("logging_strategy", "epoch")
        evaluation_strategy = configs.get("evaluation_strategy", "epoch")
        remove_unused_columns = configs.get("remove_unused_columns", False)
        output_dir = output_path
        dataloader_pin_memory = configs.get("dataloader_pin_memory", False)

        progress = kwargs.get("progress", None)
        faas_callback = kwargs.get("faas_callback", None)

        # Configure training arguments
        training_args = TrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta2=adam_beta2,
            optim=optim,
            bf16=bf16,
            remove_unused_columns=remove_unused_columns,
            save_strategy=save_strategy,
            logging_strategy=logging_strategy,
            evaluation_strategy=evaluation_strategy,
            output_dir=output_dir,
            dataloader_pin_memory=dataloader_pin_memory,
        )

        # Create save callback
        save_callback = SaveEpochCallback(
            save_path=output_path,
            tokenizer=self.processor,
            model_entity=self.model_entity,
            save_every_n_epochs=save_every_n_epochs,
            progress=progress,
            num_epochs=num_train_epochs,
            faas_callback=faas_callback,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
            data_collator=self._process_inputs,
            callbacks=[save_callback],
        )
        trainer.can_return_loss = True  # https://github.com/huggingface/peft/issues/1881

        # Train
        trainer.train()

    def prepare_data(
        self,
        dataset: dl.Dataset,
        # paths
        root_path=None,
        data_path=None,
        output_path=None,
        #
        overwrite=False,
        **kwargs,
    ):
        """
        Prepares paths for downloading dataset.

        :param dataset: dl.Dataset
        :param root_path: `str` root directory for training. default is "tmp". Can be set using self.adapter_defaults.root_path
        :param data_path: `str` dataset directory. default <root_path>/"data". Can be set using self.adapter_defaults.data_path
        :param output_path: `str` save everything to this folder. default <root_path>/"output". Can be set using self.adapter_defaults.output_path

        :param bool overwrite: overwrite the data path (download again). default is False
        """
        # define paths
        dataloop_path = dl.service_defaults.DATALOOP_PATH
        root_path = self.adapter_defaults.resolve("root_path", root_path)
        data_path = self.adapter_defaults.resolve("data_path", data_path)
        output_path = self.adapter_defaults.resolve("output_path", output_path)

        if root_path is None:
            now = datetime.now()
            root_path = os.path.join(
                dataloop_path,
                'model_data',
                "{s_id}_{s_n}".format(s_id=self.model_entity.id, s_n=self.model_entity.name),
                now.strftime('%Y-%m-%d-%H%M%S'),
            )
        if data_path is None:
            data_path = os.path.join(root_path, 'datasets', self.model_entity.dataset.id)
            os.makedirs(data_path, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(root_path, 'output')
            os.makedirs(output_path, exist_ok=True)

        self.convert_from_dtlpy(data_path, self.prompt_items_dir)

        return root_path, data_path, output_path

    def convert_from_dtlpy(self, data_path, prompt_items_dir):
        """Convert Dataloop data to format suitable for training."""
        dataset = self.model_entity.dataset
        
        # Download all items (images and prompt items)
        dataset.items.download(
            local_path=data_path, annotation_options=dl.ViewAnnotationOptions.JSON, include_annotations_in_output=True
        )
        # Export metadata/annotations
        dataset.export(local_path=data_path, include_annotations=True)

        # Inline get_entities_from_json
        entities = []
        for json_file in Path(data_path).glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                entities.extend(data)

        # Inline split_entities
        split_entities_ids = {"train": [], "validation": []}
        prompt_lookup = {}
        image_lookup = {}

        # sort through downloaded data
        for entity in entities:
            entity_filepath = entity.get("filename").split('/')[1:]
            entity_id = entity.get("id", None)
            dir_path = entity.get("dir", "")
            subset_tags = entity.get("metadata", {}).get("system", {}).get("tags", None)
            mimetype = entity.get("metadata", {}).get("system", {}).get("mimetype", "")

            # create a master lookup for prompt items and images
            if mimetype == "application/json" and dir_path.endswith(prompt_items_dir):
                prompt_path = os.path.join(data_path, "items", *entity_filepath)
                annotation_path = os.path.join(data_path, "json", *entity_filepath)
                prompt_lookup[entity_id] = {
                    "id": entity_id,
                    "prompt_path": prompt_path,
                    "annotation_path": annotation_path,
                }

                # split train and validations based on prompt items only
                if "train" in subset_tags.keys():
                    split_entities_ids["train"].append(entity_id)
                elif "validation" in subset_tags.keys():
                    split_entities_ids["validation"].append(entity_id)

            elif mimetype.startswith("image/"):  # and dir_path.endswith(ORIGINALS_DIR):
                image_path = os.path.join(data_path, "items", *entity_filepath)
                image_lookup[entity_id] = {"id": entity_id, "image_path": image_path}

        train_images, train_captions = self._get_subset_pairs(split_entities_ids["train"], prompt_lookup, image_lookup)
        validation_images, validation_captions = self._get_subset_pairs(
            split_entities_ids["validation"], prompt_lookup, image_lookup
        )

        logger.info(f"Total: {len(train_images)} images and {len(train_captions)} captions loaded")
        logger.info(f"Total: {len(validation_images)} images and {len(validation_captions)} captions loaded")

        if len(train_images) == 0 or len(validation_images) == 0:
            raise ValueError(
                f"Missing subset for training. Please check that the dataset has training and validation subsets defined on items."
            )

        self.train_dataset = Dataset.from_list(
            [{"image": img, "caption": cap} for img, cap in zip(train_images, train_captions)]
        )
        self.val_dataset = Dataset.from_list(
            [{"image": img, "caption": cap} for img, cap in zip(validation_images, validation_captions)]
        )

        return self.train_dataset, self.val_dataset

    @staticmethod
    def _get_subset_pairs(entities_ids, prompt_lookup, image_lookup):
        """Get subset pairs for all prompt items"""
        images = []
        captions = []
        for entity_id in entities_ids:
            prompt_file_path = prompt_lookup.get(entity_id).get("prompt_path")
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
            stream_url = prompt_data.get("prompts", {"1": [{"value": None}]}).get("1")[0].get("value")
            original_item_id = None
            if stream_url:
                parts = stream_url.split("/")
                for i, part in enumerate(parts):
                    if part == "items" and i + 1 < len(parts):
                        original_item_id = parts[i + 1]
                        break
            image = None
            item_image_path = image_lookup.get(original_item_id, {}).get("image_path")
            if item_image_path and os.path.exists(item_image_path):
                image = Image.open(item_image_path)
            if image is not None:
                annotation_file_path = prompt_lookup.get(entity_id).get("annotation_path")
                caption = ""
                if os.path.exists(annotation_file_path):
                    with open(annotation_file_path, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        annotations = annotation_data.get("annotations", [])
                        for ann in annotations:
                            if ann.get("coordinates"):
                                caption = ann.get("coordinates")
                                break
                images.append(image)
                captions.append(caption)
        return images, captions

    def _prepare_model_and_tokenizer(self, ckpt=None):
        """Prepare the vision-language model and tokenizer for predicting or training LoRA."""
        use_lora = self.configuration.get("use_lora", True)
        freeze_llm = self.configuration.get("freeze_llm", False)
        freeze_image = self.configuration.get("freeze_image", False)

        if use_lora is True:
            lora_config = LoraConfig(
                r=self.configuration.get("r", 8),
                lora_alpha=self.configuration.get("lora_alpha", 8),
                lora_dropout=self.configuration.get("lora_dropout", 0.1),
                target_modules=self.configuration.get(
                    "target_modules", ["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"]
                ),
                use_dora=self.configuration.get("use_dora", True),  # optional DoRA
                init_lora_weights=self.configuration.get("init_lora_weights", "gaussian"),
            )

            # Initialize model with proper device handling
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                low_cpu_mem_usage=True,  # Enable low CPU memory usage
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        elif freeze_image is True:
            if freeze_llm is True:
                raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device_map
            )
            # freeze vision model to save up on compute
            for param in model.vision_model.parameters():
                param.requires_grad = False

        elif freeze_llm is True:
            if freeze_image is True:
                raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device_map
            )
            # freeze text model, this is encouraged in paper
            for param in model.language_model.parameters():
                param.requires_grad = False

        else:  # full ft
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device_map
            )

        processor = AutoProcessor.from_pretrained(ckpt)

        return model, processor

    def _process_inputs(self, examples):
        """Prepare items for training or inference."""
        texts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Answer briefly. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['caption']}<|eot_id|>"
            for example in examples
        ]
        images = [[example["image"].convert("RGB")] for example in examples]

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == 128256] = -100  # image token index
        batch["labels"] = labels
        batch = batch.to(torch.bfloat16).to(self.device)

        return batch

    @staticmethod
    def _reformat_messages(messages, system_prompt=None):
        # In case of multiple messages,
        # we assume the last user message contains the image of interest

        last_user_message = ModelAdapter._get_last_prompt_message(messages)

        prompt_txt = None
        image = None

        # The last user message may contain multiple contents,
        # such as a text component and an image component
        # or multiple text components (e.g., multiple questions)
        for content in last_user_message["content"]:

            content_type = content.get("type", None)
            if content_type is None:
                raise ValueError("Message content type not found")

            if content_type == "text":
                # Concatenate multiple text contents with space
                new_text = content.get("text", "").strip()
                if new_text:
                    if prompt_txt is None:
                        prompt_txt = new_text
                    else:
                        prompt_txt = f"{prompt_txt} {new_text}".strip()

            elif content_type == "image_url":
                image_url = content.get("image_url", {}).get("url")
                if image_url is not None:
                    if image is not None:  # i.e., we previously found an image
                        raise ValueError("Multiple images not supported")
                    else:
                        base64_str = content["image_url"]["url"].split("base64,")[1]
                        image_buffer = BytesIO(base64.b64decode(base64_str))
                        image = Image.open(image_buffer).convert("RGB")
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        if prompt_txt is None:
            if system_prompt is not None:
                prompt_txt = system_prompt
            else:
                prompt_txt = "What is in this image?"
        prompt_txt = "Question: {} Answer:".format(prompt_txt)

        if image is None:
            raise ValueError("No image found in messages.")

        return prompt_txt, image

    @staticmethod
    def _get_last_prompt_message(messages):
        text = None
        try:
            for message in reversed(messages):
                if message.get("role") == "user":
                    text = message
        except Exception as e:
            logger.error(f"Error getting last message from messages: {e}. Skipping...")
        return text


class SaveEpochCallback(TrainerCallback):
    """Custom callback to save model after each epoch and track eval loss."""

    def __init__(self, save_path, tokenizer, model_entity, progress, num_epochs, faas_callback, save_every_n_epochs):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.model_entity = model_entity
        self.save_every_n_epochs = save_every_n_epochs
        self.progress = progress
        self.num_epochs = num_epochs
        self.faas_callback = faas_callback

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.current_loss = None
        self.current_epoch = 0
        self.eval_losses = []  # Track evaluation losses
        self.log_file = os.path.join(self.save_path, "training_logs.json")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.current_loss = logs["loss"]
            self.current_epoch = logs["epoch"]
            if self.best_loss == float("inf"):
                self.best_loss = self.current_loss
                logger.info(f"Initial best loss set to: {self.best_loss:.4f}")
            self.model_entity.metrics.create(
                samples=dl.PlotSample(figure="loss", legend="train", x=self.current_epoch, y=self.current_loss),
                dataset_id=self.model_entity.dataset_id,
            )
            current_logs = state.log_history
            with open(self.log_file, "w") as f:
                json.dump(current_logs, f, indent=2)
            logger.info(f"Updated training logs in {self.log_file}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            # epoch_loss = {"epoch": self.current_epoch, "eval_loss": eval_loss}
            self.eval_losses.append(eval_loss)

            # Log the evaluation loss
            logger.info(f"Epoch {self.current_epoch:.2f} - Eval Loss: {eval_loss:.4f}")

            # Add eval loss to the metrics visualization
            self.model_entity.metrics.create(
                samples=dl.PlotSample(figure="loss", legend="validation", x=self.current_epoch, y=eval_loss),
                dataset_id=self.model_entity.dataset_id,
            )

            # # Consider using eval loss instead of training loss for model selection
            # if eval_loss < self.best_loss:
            #     self.best_loss = eval_loss
            #     self.best_epoch = self.current_epoch
            #     logger.info(f"New best model based on eval loss: {eval_loss:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError("Cannot save model. Model is None.")
        if self.current_epoch > 0:
            if self.current_loss is not None and self.current_loss < self.best_loss:
                logger.info(f"\nNew best model found! Loss updated to {self.current_loss:.4f}")
                self.best_loss = self.current_loss
                self.best_epoch = self.current_epoch
                self.current_loss = None
                best_model_dir = os.path.join(self.save_path, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                logger.info(f"Best model saved in {best_model_dir}")
            else:
                logger.info(
                    f"Skipping save for epoch {self.current_epoch} (Current loss: {self.current_loss:.4f}, Best loss: {self.best_loss:.4f})"
                )

        if self.progress is not None:
            if self.faas_callback is not None:
                self.faas_callback(self.current_epoch, self.num_epochs)
                