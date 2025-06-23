import json
import os
from pathlib import Path
import PIL
import torch
import logging
import dtlpy as dl
from typing import List
from transformers import DFineForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_callback import TrainerState, TrainerControl
import shutil
from dtlpyconverters import services, coco_converters
from pycocotools.coco import COCO
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets
from PIL import Image
from datasets import Dataset, concatenate_datasets, load_from_disk
from PIL import Image
import torch
import os

# installs :
# pip install dtlpy
# pip install datasets
# pip install git+https://github.com/dataloop-ai-apps/dtlpy-converters
# pip install -U transformers
# pip install python-dotenv
#
# pip install async-

logger = logging.getLogger("[D-FINE]")

class HuggingAdapter(dl.BaseModelAdapter):
    class EpochEndCallback(TrainerCallback):
    def __init__(self, model_adapter: HuggingAdapter,faas_callback: Optional[Callable] = None):
        super().__init__()
        self.model_adapter = model_adapter
        self.faas_callback = faas_callback

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("on_epoch_end")
        print(f"✅ Epoch {state.epoch} ended")
        print(f"state: {state}")
        # You can add more logic here (logging, saving, eval, etc.)
        if self.faas_callback:
            self.faas_callback(state.epoch,self.model_adapter.configuration['train_configs']['num_train_epochs'])
        self.configuration['start_epoch'] = state.epoch + 1
        return control
    
    @staticmethod
    def _process_coco_json(output_annotations_path: str) -> None:
        """
        Process COCO JSON annotations file to make it compatible with RF-DETR requirements.

        RF-DETR requires integer IDs and supercategory fields for categories. This function converts
        string IDs to integers via hashing, adds missing supercategory fields, and updates file paths
        to match new image locations by keeping only filenames.

        Args:
            output_annotations_path (str): Path to directory containing the COCO JSON file

        Returns:
            None
        """
        src_json_path = os.path.join(output_annotations_path, 'coco.json')
        dest_json_path = os.path.join(output_annotations_path, '_annotations.coco.json')

        logger.info(f'Processing COCO JSON file at {src_json_path}')
        # Load the JSON file
        with open(src_json_path, 'r') as f:
            coco_data = json.load(f)

        # Add supercategory field to each category if it doesn't exist
        for category in coco_data.get('categories', []):
            if 'supercategory' not in category:
                category['supercategory'] = 'none'

        # Convert image IDs to integers and clean file names
        for image in coco_data.get('images', []):
            if isinstance(image['id'], str):
                image['id'] = abs(hash(image['id']))
            # Remove parent directory from file_name
            if '/' in image['file_name']:
                image['file_name'] = os.path.basename(image['file_name'])

        # Convert annotation IDs and image_ids to integers
        for annotation in coco_data.get('annotations', []):
            if isinstance(annotation['id'], str):
                annotation['id'] = abs(hash(annotation['id']))
            if isinstance(annotation['image_id'], str):
                annotation['image_id'] = abs(hash(annotation['image_id']))

        with open(dest_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info('COCO JSON processing completed')

    def _get_hugging_dataset(self, data_path: str) -> tuple[Dataset, Dataset]:
        logger.info('get_hugging_dataset')

        def load_coco_as_list(annotation_path, image_dir):
            coco = COCO(annotation_path)
            items = []
            for image_id in coco.imgs:
                image_info = coco.loadImgs(image_id)[0]
                file_path = os.path.join(image_dir, image_info["file_name"])
                anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

                boxes = []
                class_labels = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    class_labels.append(ann["category_id"])

                items.append({"image": file_path, "class_labels": class_labels, "boxes": boxes})
            return items

        # Step 2: Prepare datasets
        logger.info('load_coco_as_list train')
        train_dataset = Dataset.from_list(
            load_coco_as_list(
                os.path.join(data_path, "train", "_annotations.coco.json"), os.path.join(data_path, "train")
            )
        )
        logger.info('load_coco_as_list val')
        val_dataset = Dataset.from_list(
            load_coco_as_list(
                os.path.join(data_path, "valid", "_annotations.coco.json"), os.path.join(data_path, "valid")
            )
        )

        # ------------------------- 1. preprocessing -------------------------
        def preprocess(example):
            return {"image_path": example["image"], "class_labels": example["class_labels"], "boxes": example["boxes"]}

        logger.info('datasets map')
        # Step 4: Map preprocessing
        train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)  # drop original cols
        val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)
        logger.info('datasets map done')
        return train_dataset, val_dataset

    def load(self, local_path, **kwargs):
        logger.info(f"Loading model from {local_path}")
        self.model_name = self.configuration.get("model_name", "d-fine")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_processor_path = self.configuration.get("image_processor_path", "ustc-community/dfine-xlarge-obj2coco")
        self.processor = AutoImageProcessor.from_pretrained(image_processor_path)

        self.model = None
        checkpoint_path = self.configuration.get("checkpoint_path", "ustc-community/dfine-xlarge-obj2coco")
        if checkpoint_path != "" and checkpoint_path.strip() != "":
            required_files = ['config.json', 'model.safetensors', 'training_args.bin', 'trainer_state.json']
            has_required_files = all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_files)
            if not has_required_files:
                logger.warning(f"Checkpoint path {checkpoint_path} does not contain required files: {required_files}")
            else:
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")
                self.model = DFineForObjectDetection.from_pretrained(
                    pretrained_model_name_or_path=checkpoint_path, local_files_only=True, use_safetensors=True
                )
        if self.model is None:
            self.model = DFineForObjectDetection.from_pretrained(checkpoint_path)
        self.model.to(self.device)

    def predict(self, batch: List[dl.Item], **kwargs):
        batch_annotations = []

        for item in batch:
            try:
                # Download and process image
                image_buffer = item.download(save_locally=False)
                image = PIL.Image.open(image_buffer).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                # Perform inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process results
                target_sizes = torch.tensor([image.size[::-1]])
                results = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.3
                )[0]

                # Create box annotations for each detection
                item_annotations = dl.AnnotationCollection()
                for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                    score, label = score.item(), label_id.item()
                    box = [round(i, 2) for i in box.tolist()]
                    left, top, right, bottom = box

                    # Create box annotation
                    item_annotations.add(
                        annotation_definition=dl.Box(
                            label=self.model.config.id2label[label], top=top, left=left, bottom=bottom, right=right
                        ),
                        model_info={
                            "name": self.model_name,
                            "model_id": self.model_name,
                            "confidence": round(score, 3),
                        },
                    )
                batch_annotations.append(item_annotations)

            except Exception as e:
                logger.error(f"Error processing item {item.id}: {str(e)}")

        return batch_annotations

    def save(self, local_path: str, **kwargs) -> None:
        """
        Save model configuration by updating the weights filename.

        This method updates the model configuration to point to the best checkpoint weights file.

        Args:
            local_path (str): Path where model files are saved (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            None
        """
        self.configuration.update({'weights_filename': 'checkpoint_best_total.pth'})

    def convert_from_dtlpy(self, data_path: str, **kwargs) -> None:
        """Convert dataset from Dataloop format to COCO format.

        This method converts a Dataloop dataset to COCO format required by RF-DETR. It validates box annotations
        in each subset (train/validation) and converts them to match RF-DETR's train/valid directory structure.

        Args:
            data_path (str): Path to the directory where the dataset will be converted
            **kwargs: Additional keyword arguments (unused)

        Raises:
            ValueError: If model has no labels defined or if no box annotations are found in a subset
        """
        logger.info(f'Converting dataset from Dataloop format to COCO format at {data_path}')

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if len(self.model_entity.labels) == 0:
            logger.error("Model has no labels defined")
            raise ValueError('model.labels is empty. Model entity must have labels')

        for subset_name in subsets.keys():
            logger.info(f'Converting subset: {subset_name} to COCO format')

            # rf-detr expects train and valid folders
            dist_dir_name = subset_name if subset_name != 'validation' else 'valid'
            input_annotations_path = os.path.join(data_path, subset_name, 'json')
            output_annotations_path = os.path.join(data_path, dist_dir_name)

            # Ensure instance map IDs start from 1 not 0
            if 0 in self.model_entity.dataset.instance_map.values():
                self.model_entity.dataset.instance_map = {
                    label: label_id + 1 for label, label_id in self.model_entity.dataset.instance_map.items()
                }

            converter = coco_converters.DataloopToCoco(
                output_annotations_path=output_annotations_path,
                input_annotations_path=input_annotations_path,
                download_items=False,
                download_annotations=False,
                dataset=self.model_entity.dataset,
            )

            coco_converter_services = services.converters_service.DataloopConverters()
            loop = coco_converter_services._get_event_loop()
            try:
                loop.run_until_complete(converter.convert_dataset())
            except Exception as e:
                raise Exception(f"Error converting subset {subset_name}: {str(e)}")

            # convert coco.json to _annotations.coco.json
            HuggingAdapter._process_coco_json(output_annotations_path)

            # Move images from the deepest subdirectory under <data_path>/<subset_name>/items/
            # to <data_path>/<dist_dir_name>. For example:
            # From: <data_path>/train/items/dir1/dir2/images.jpg
            # To:   <data_path>/train/images.jpg
            src_parent_dir = os.path.join(data_path, subset_name)
            dst_images_path = os.path.join(data_path, dist_dir_name)

            # Moving files to a temporary directory first because in most cases source and destination
            # directories have the same name (e.g. 'train' -> 'train'). Direct move would fail.
            tmp_dir_path = os.path.join(data_path, f'tmp_dir_{subset_name}')
            os.makedirs(tmp_dir_path)
            json_file_path = os.path.join(output_annotations_path, '_annotations.coco.json')
            shutil.move(json_file_path, tmp_dir_path)

            # move all non json files to tmp_dir_path
            all_files = [f for f in Path(src_parent_dir).rglob('*') if f.is_file() and not f.name.endswith('.json')]
            if len(all_files) == 0:
                raise FileNotFoundError(f"No files found in {src_parent_dir}")
            for file in all_files:
                shutil.move(file, tmp_dir_path)

            # Remove the original subset directory
            logger.info(f'Removing {os.path.join(data_path, subset_name)}')
            shutil.rmtree(os.path.join(data_path, subset_name))
            if os.path.exists(dst_images_path):
                shutil.rmtree(os.path.join(dst_images_path))

            logger.info(f'Moving directory from {tmp_dir_path} to {dst_images_path}')
            shutil.move(tmp_dir_path, dst_images_path)

    def get_training_args(self, output_path: str):
        train_config_dict = self.configuration.get('train_configs', {})
        logger.info(f'train_config_dict: {train_config_dict}')
        return TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=train_config_dict.get('overwrite_output_dir', False),
            do_train=train_config_dict.get('do_train', False),
            do_eval=train_config_dict.get('do_eval', False),
            do_predict=train_config_dict.get('do_predict', False),
            eval_strategy=train_config_dict.get('eval_strategy', "epoch"),
            prediction_loss_only=train_config_dict.get('prediction_loss_only', False),
            per_device_train_batch_size=train_config_dict.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=train_config_dict.get('per_device_eval_batch_size', 8),
            per_gpu_train_batch_size=train_config_dict.get('per_gpu_train_batch_size', None),
            per_gpu_eval_batch_size=train_config_dict.get('per_gpu_eval_batch_size', None),
            gradient_accumulation_steps=train_config_dict.get('gradient_accumulation_steps', 1),
            eval_accumulation_steps=train_config_dict.get('eval_accumulation_steps', None),
            eval_delay=train_config_dict.get('eval_delay', 0),
            torch_empty_cache_steps=train_config_dict.get('torch_empty_cache_steps', None),
            learning_rate=train_config_dict.get('learning_rate', 5e-5),
            weight_decay=train_config_dict.get('weight_decay', 0.0),
            adam_beta1=train_config_dict.get('adam_beta1', 0.9),
            adam_beta2=train_config_dict.get('adam_beta2', 0.999),
            adam_epsilon=train_config_dict.get('adam_epsilon', 1e-8),
            max_grad_norm=train_config_dict.get('max_grad_norm', 1.0),
            num_train_epochs=train_config_dict.get('num_train_epochs', 3.0),
            max_steps=train_config_dict.get('max_steps', -1),
            lr_scheduler_type=train_config_dict.get('lr_scheduler_type', "linear"),
            lr_scheduler_kwargs=train_config_dict.get('lr_scheduler_kwargs', {}),
            warmup_ratio=train_config_dict.get('warmup_ratio', 0.0),
            warmup_steps=train_config_dict.get('warmup_steps', 0),
            log_level=train_config_dict.get('log_level', "passive"),
            log_level_replica=train_config_dict.get('log_level_replica', "warning"),
            log_on_each_node=train_config_dict.get('log_on_each_node', True),
            logging_dir=train_config_dict.get('logging_dir', None),
            logging_strategy=train_config_dict.get('logging_strategy', "steps"),
            logging_first_step=train_config_dict.get('logging_first_step', False),
            logging_steps=train_config_dict.get('logging_steps', 500),
            logging_nan_inf_filter=train_config_dict.get('logging_nan_inf_filter', True),
            save_strategy=train_config_dict.get('save_strategy', "epoch"),
            save_steps=train_config_dict.get('save_steps', 500),
            save_total_limit=train_config_dict.get('save_total_limit', None),
            save_safetensors=train_config_dict.get('save_safetensors', True),
            save_on_each_node=train_config_dict.get('save_on_each_node', False),
            save_only_model=train_config_dict.get('save_only_model', False),
            restore_callback_states_from_checkpoint=train_config_dict.get(
                'restore_callback_states_from_checkpoint', False
            ),
            no_cuda=train_config_dict.get('no_cuda', False),
            use_cpu=train_config_dict.get('use_cpu', False),
            use_mps_device=train_config_dict.get('use_mps_device', False),
            seed=train_config_dict.get('seed', 42),
            data_seed=train_config_dict.get('data_seed', None),
            jit_mode_eval=train_config_dict.get('jit_mode_eval', False),
            use_ipex=train_config_dict.get('use_ipex', False),
            bf16=train_config_dict.get('bf16', False),
            fp16=train_config_dict.get('fp16', False),
            fp16_opt_level=train_config_dict.get('fp16_opt_level', "O1"),
            half_precision_backend=train_config_dict.get('half_precision_backend', "auto"),
            bf16_full_eval=train_config_dict.get('bf16_full_eval', False),
            fp16_full_eval=train_config_dict.get('fp16_full_eval', False),
            tf32=train_config_dict.get('tf32', None),
            local_rank=train_config_dict.get('local_rank', -1),
            ddp_backend=train_config_dict.get('ddp_backend', None),
            tpu_num_cores=train_config_dict.get('tpu_num_cores', None),
            tpu_metrics_debug=train_config_dict.get('tpu_metrics_debug', False),
            debug=train_config_dict.get('debug', ""),
            dataloader_drop_last=train_config_dict.get('dataloader_drop_last', False),
            eval_steps=train_config_dict.get('eval_steps', None),
            dataloader_num_workers=train_config_dict.get('dataloader_num_workers', 0),
            dataloader_prefetch_factor=train_config_dict.get('dataloader_prefetch_factor', None),
            past_index=train_config_dict.get('past_index', -1),
            run_name=train_config_dict.get('run_name', None),
            disable_tqdm=train_config_dict.get('disable_tqdm', None),
            remove_unused_columns=False,
            label_names=train_config_dict.get('label_names', None),
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            ignore_data_skip=train_config_dict.get('ignore_data_skip', False),
            fsdp=train_config_dict.get('fsdp', ""),
            fsdp_min_num_params=train_config_dict.get('fsdp_min_num_params', 0),
            fsdp_config=train_config_dict.get('fsdp_config', None),
            fsdp_transformer_layer_cls_to_wrap=train_config_dict.get('fsdp_transformer_layer_cls_to_wrap', None),
            accelerator_config=train_config_dict.get('accelerator_config', None),
            deepspeed=train_config_dict.get('deepspeed', None),
            label_smoothing_factor=train_config_dict.get('label_smoothing_factor', 0.0),
            optim=train_config_dict.get('optim', "adamw_torch"),
            optim_args=train_config_dict.get('optim_args', None),
            adafactor=train_config_dict.get('adafactor', False),
            group_by_length=train_config_dict.get('group_by_length', False),
            length_column_name=train_config_dict.get('length_column_name', "length"),
            report_to=train_config_dict.get('report_to', None),
            ddp_find_unused_parameters=train_config_dict.get('ddp_find_unused_parameters', None),
            ddp_bucket_cap_mb=train_config_dict.get('ddp_bucket_cap_mb', None),
            ddp_broadcast_buffers=train_config_dict.get('ddp_broadcast_buffers', None),
            dataloader_pin_memory=train_config_dict.get('dataloader_pin_memory', True),
            dataloader_persistent_workers=train_config_dict.get('dataloader_persistent_workers', False),
            skip_memory_metrics=train_config_dict.get('skip_memory_metrics', True),
            use_legacy_prediction_loop=train_config_dict.get('use_legacy_prediction_loop', False),
            push_to_hub=train_config_dict.get('push_to_hub', False),
            resume_from_checkpoint=train_config_dict.get('resume_from_checkpoint', None),
            hub_model_id=train_config_dict.get('hub_model_id', None),
            hub_strategy=train_config_dict.get('hub_strategy', "every_save"),
            hub_token=train_config_dict.get('hub_token', None),
            hub_private_repo=train_config_dict.get('hub_private_repo', None),
            hub_always_push=train_config_dict.get('hub_always_push', False),
            gradient_checkpointing=train_config_dict.get('gradient_checkpointing', False),
            gradient_checkpointing_kwargs=train_config_dict.get('gradient_checkpointing_kwargs', None),
            include_inputs_for_metrics=train_config_dict.get('include_inputs_for_metrics', False),
            include_for_metrics=train_config_dict.get('include_for_metrics', []),
            eval_do_concat_batches=train_config_dict.get('eval_do_concat_batches', True),
            fp16_backend=train_config_dict.get('fp16_backend', "auto"),
            push_to_hub_model_id=train_config_dict.get('push_to_hub_model_id', None),
            push_to_hub_organization=train_config_dict.get('push_to_hub_organization', None),
            push_to_hub_token=train_config_dict.get('push_to_hub_token', None),
            auto_find_batch_size=train_config_dict.get('auto_find_batch_size', False),
            full_determinism=train_config_dict.get('full_determinism', False),
            torchdynamo=train_config_dict.get('torchdynamo', None),
            ray_scope=train_config_dict.get('ray_scope', "last"),
            ddp_timeout=train_config_dict.get('ddp_timeout', 1800),
            torch_compile=train_config_dict.get('torch_compile', False),
            torch_compile_backend=train_config_dict.get('torch_compile_backend', None),
            torch_compile_mode=train_config_dict.get('torch_compile_mode', None),
            include_tokens_per_second=train_config_dict.get('include_tokens_per_second', False),
            include_num_input_tokens_seen=train_config_dict.get('include_num_input_tokens_seen', False),
            neftune_noise_alpha=train_config_dict.get('neftune_noise_alpha', None),
            optim_target_modules=train_config_dict.get('optim_target_modules', None),
            batch_eval_metrics=train_config_dict.get('batch_eval_metrics', False),
            eval_on_start=train_config_dict.get('eval_on_start', False),
            use_liger_kernel=train_config_dict.get('use_liger_kernel', False),
            eval_use_gather_object=train_config_dict.get('eval_use_gather_object', False),
            average_tokens_across_devices=train_config_dict.get('average_tokens_across_devices', False),
        )

    def save(self, local_path: str, **kwargs) -> None:
        pass

    def train(self, data_path: str, output_path: str, **kwargs) -> None:
        train_dataset, val_dataset = self._get_hugging_dataset(data_path)

        # Step 5: Collate function
        # ------------------------- 3. custom collate_fn -------------------------
        def collate_fn(batch):
            images = [Image.open(example["image_path"]).convert("RGB") for example in batch]
            encodings = [self.processor(images=img, return_tensors="pt") for img in images]
            pixel_values = torch.stack([enc["pixel_values"].squeeze(0) for enc in encodings])

            labels = [
                {
                    "class_labels": torch.tensor(b["class_labels"], dtype=torch.long),
                    "boxes": torch.tensor(b["boxes"], dtype=torch.float),
                }
                for b in batch
            ]
            return {"pixel_values": pixel_values, "labels": labels}

        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(output_path),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            callbacks=[EpochEndCallback()],
        )
        logger.info("start real training")
        trainer.train()
        logger.info("training done")


if __name__ == "__main__":
    print("start")
    use_rc_env = False
    if use_rc_env:
        dl.setenv('rc')
    else:
        dl.setenv('prod')
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('DTLPY_API_KEY')
    print(f"api_key: {api_key}")
    if api_key:
        print("DTLPY_API_KEY found in environment variables")
        dl.login_api_key(api_key=api_key)
    else:
        print("ERROR: DTLPY_API_KEY not found in environment variables")
        raise ValueError("Missing required DTLPY_API_KEY environment variable")

    if dl.token_expired():
        dl.login()
    print("login done")
    if use_rc_env:
        project = dl.projects.get(project_name='Husam Testing')
    else:
        project = dl.projects.get(project_name='ShadiDemo')
        # project = dl.projects.get(project_name='IPM development')
    print("project done")
    # model = project.models.get(model_name='rd-dert-used-for-dfine-train-hfg')
    model = project.models.get(model_name='rf-detr-sdk-clone-1')
    print("model done")
    model.status = 'pre-trained'
    model_adapter = HuggingAdapter(model)
    print("model_adapter done - start train")
    model_adapter.train_model(model=model)
    print("convert done")
