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

        # load base model
        logger.info(f"Downloading model from HuggingFace {hf_model_name}")
        base_model, self.processor = self._prepare_model_and_tokenizer(ckpt=hf_model_name)

        # check if lora adapter weights exist
        lora_dir = os.path.join(local_path, "lora_weights")  # TODO check saving works
        if os.path.exists(lora_dir) is True:
            logger.info(f"Loading LoRA weights from {lora_dir}")
            model_to_merge = PeftModel.from_pretrained(base_model, lora_dir, device_map=self.device)
            merged_model = model_to_merge.merge_and_unload()
            self.model = merged_model
        else:
            self.model = base_model

    def save(self, local_path, **kwargs):
        """Save the model and processor to Dataloop"""
        save_dir = os.path.join(local_path, "lora_weights")
        self.model.save_pretrained(save_directory=save_dir)
        self.processor.save_pretrained(save_directory=save_dir)
        logger.info(f"Successfully saved trained LoRA weights to {save_dir}")

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
            progress=progress,
            num_epochs=num_train_epochs,
        )

        # Get training data
        train_dataset, val_dataset = self.convert_from_dtlpy(data_path)

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            data_collator=self._process_inputs,
            callbacks=[save_callback],
        )
        trainer.can_return_loss = True  # https://github.com/huggingface/peft/issues/1881

        # Train
        trainer.train()

    def convert_from_dtlpy(self, data_path, **kwargs):
        """Convert Dataloop data to format suitable for training."""
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", {}).get("subsets", None)
        if "train" not in subsets:
            raise ValueError(
                "Could not find train set. Llama Vision requires train and validation set for training. "
                "Add a train set DQL filter in the dl.Model metadata"
            )
        if "validation" not in subsets:
            raise ValueError(
                "Could not find validation set. Llama Vision requires train and validation set for training. "
                "Add a validation set DQL filter in the dl.Model metadata"
            )

        for subset, filters_dict in subsets.items():
            data_subset_base_path = os.path.join(data_path, subset)
            # Add json type validation
            new_condition = {"metadata.system.mimetype": {"$eq": "application/json"}}
            if new_condition not in filters_dict["filter"]["$and"]:
                filters_dict["filter"]["$and"].append(new_condition)

            filters = dl.Filters(custom_filter=filters_dict)
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f"Finetune Datasets expects only json files in the subset {subset}. Found 0 jsons files in subset {subset}."
                )

            # Items are not downloaded in prepare_data() because of the annotations filters
            self.model_entity.dataset.items.download(filters=filters, local_path=data_subset_base_path)

        # Get image-text pairs for training and validation
        train_items, train_captions = self._get_img_txt_pairs(os.path.join(data_path, "train"), overwrite=False)
        val_items, val_captions = self._get_img_txt_pairs(os.path.join(data_path, "validation"), overwrite=False)

        logger.info(f"number of train items: {len(train_items)}")
        logger.info(f"number of val items: {len(val_items)}")

        # Convert to Dataset objects
        train_dataset = Dataset.from_list(
            [{"image": img, "caption": cap} for img, cap in zip(train_items, train_captions)]
        )
        val_dataset = Dataset.from_list([{"image": img, "caption": cap} for img, cap in zip(val_items, val_captions)])

        return train_dataset, val_dataset

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
                device_map=self.device,
                low_cpu_mem_usage=True,  # Enable low CPU memory usage
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        elif freeze_image is True:
            if freeze_llm is True:
                raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device
            )
            # freeze vision model to save up on compute
            for param in model.vision_model.parameters():
                param.requires_grad = False

        elif freeze_llm is True:
            if freeze_image is True:
                raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device
            )
            # freeze text model, this is encouraged in paper
            for param in model.language_model.parameters():
                param.requires_grad = False

        else:  # full ft
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map=self.device
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
        batch = batch.to(torch.bfloat16).to("cuda")

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

    @staticmethod
    def _get_img_txt_pairs(data_path, overwrite=False):
        """Get image-text pairs from downloaded Dataloop items."""
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)

        # List all downloaded prompt item jsons and download images from link
        item_jsons = (path / "items").rglob("*.json")
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(lambda item_file: ModelAdapter._load_stream(item_file, overwrite), item_jsons))
        # image_paths = []  # DEBUG
        # for item_file in item_jsons:
        #     image_paths.append(_download_stream(item_file, overwrite))

        item_captions = []
        annots_files = (path / "json").rglob("*.json")
        for src_file in annots_files:
            with open(src_file, "r") as f:
                data = json.load(f)
            if len(data["annotations"]) > 0:
                annot = data["annotations"][0]
                if annot["label"] == "free-text":
                    item_captions.append(annot.get("coordinates", ""))
                else:
                    raise TypeError(
                        f"No free-text annotation found in json file {src_file}. Please check annotation type."
                    )
            else:
                raise ValueError(f"No annotations found in json file {src_file} to use as image caption.")

        # Clean up empty directories
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

        return images, item_captions

    @staticmethod
    def _load_stream(item_file, overwrite=False):
        """Download image from Dataloop item."""
        with open(item_file) as json_data:
            d = json.load(json_data)
        img_prompt = next(iter(d["prompts"].values()))

        item_url = None
        for dictionary in img_prompt:
            if dictionary.get("mimetype") == "image/*":
                item_url = dictionary.get("value")
                break
        if item_url is None:
            raise ValueError(f"Image URL not found in prompt item {Path(item_file).name}.")

        item_id = item_url.split("/")[-2]
        item = dl.items.get(item_id=item_id)
        download_path = item.download(local_path=Path(item_file).parents[0])
        image_name = Path(item_file).stem + Path(download_path).suffix
        new_path = Path(item_file).parents[0] / image_name

        try:
            os.rename(Path(download_path), new_path)
        except FileExistsError:
            if overwrite is True:
                logger.debug(f"Overwriting file {new_path}.")
                os.remove(new_path)
                os.rename(Path(download_path), new_path)
            else:
                logger.debug(f"File {new_path} already exists. Skipping.")

        new_image = Image.open(new_path).convert("RGB")
        return new_image


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
        progress = kwargs.get("progress", None)
        faas_callback = kwargs.get("faas_callback", None)
        num_epochs = kwargs.get("num_epochs", None)

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

        if progress is not None:
            if faas_callback is not None:
                faas_callback(self.current_epoch, num_epochs)
