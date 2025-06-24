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
from typing import Optional, Callable
import numpy as np

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
        def __init__(self, model_adapter: 'HuggingAdapter', faas_callback: Optional[Callable] = None):
            super().__init__()
            self.model_adapter = model_adapter
            self.faas_callback = faas_callback

        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            current_epoch = int(state.epoch)
            total_epochs = int(args.num_train_epochs)
            logger.info(f"Epoch {current_epoch} ended")

            # FaaS callback
            if self.faas_callback:
                self.faas_callback(current_epoch, total_epochs)

            # Collect metrics (dummy example: extract loss, adapt if more are needed)
            samples = []
            NaN_defaults = {'loss': 1.0}
            metrics = state.log_history[-1] if state.log_history else {}

            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    logger.warning(f"Non-finite value for {metric_name}. Replacing with default.")
                    value = NaN_defaults.get(metric_name, 0)

                samples.append(dl.PlotSample(figure=metric_name, legend='metrics', x=current_epoch, y=value))

            try:
                if samples:
                    self.model_adapter.model_entity.metrics.create(
                        samples=samples, dataset_id=self.model_adapter.model_entity.dataset_id
                    )
            except Exception as e:
                logger.error(f"Failed to store metrics in Dataloop: {e}")

            # Save model
            logger.info("Saving model checkpoint to model entity...")
            try:
                self.model_adapter.save_to_model(local_path=args.output_dir, cleanup=False)
            except Exception as e:
                logger.error(f"Error during model saving: {e}")

            # Update internal configuration
            self.model_adapter.configuration['start_epoch'] = current_epoch + 1
            logger.info(f"best_model_checkpoint: {state.best_model_checkpoint}")
            if state.best_model_checkpoint:
                self.model_adapter.configuration['checkpoint_path'] = os.path.basename(state.best_model_checkpoint)
            else:
                logger.info("No best model checkpoint available yet")
            self.model_adapter.model_entity.update()

            # Clean up old checkpoints, keeping only latest and best
            try:
                checkpoint_dir = args.output_dir
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
                if len(checkpoints) > 2:  # Only clean if we have more than 2 checkpoints
                    checkpoints.sort(key=lambda x: int(x.split('-')[1]))  # Sort by checkpoint number
                    latest_checkpoint = checkpoints[-1]
                    best_checkpoint = (
                        os.path.basename(state.best_model_checkpoint) if state.best_model_checkpoint else None
                    )

                    for checkpoint in checkpoints:
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                        if checkpoint != latest_checkpoint and checkpoint != best_checkpoint:
                            logger.info(f"Removing old checkpoint: {checkpoint}")
                            shutil.rmtree(checkpoint_path)
            except Exception as e:
                logger.error(f"Error cleaning up old checkpoints: {e}")
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

    def get_training_args(self, output_path: str) -> TrainingArguments:
        cfg = self.configuration.get('train_configs', {})
        logger.info(f'train_config_dict: {cfg}')

        return TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=cfg.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=cfg.get('per_device_eval_batch_size', 8),
            learning_rate=cfg.get('learning_rate', 5e-5),
            weight_decay=cfg.get('weight_decay', 0.0),
            num_train_epochs=cfg.get('num_train_epochs', 3),
            logging_strategy="epoch",
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=cfg.get('save_total_limit', 2),
            load_best_model_at_end=True,
            metric_for_best_model=cfg.get('metric_for_best_model', "loss"),
            greater_is_better=cfg.get('greater_is_better', False),
            resume_from_checkpoint=cfg.get('resume_from_checkpoint', None),
            remove_unused_columns=False,
            fp16=cfg.get('fp16', False),
        )

    def save(self, local_path: str, **kwargs) -> None:
        pass

    def train(self, data_path: str, output_path: str, **kwargs) -> None:
        train_dataset, val_dataset = self._get_hugging_dataset(data_path)

        # Resume from checkpoint logic
        start_epoch = self.configuration.get('start_epoch', 0)
        checkpoint_path = self.configuration.get('checkpoint_path', None)

        resume_checkpoint = None
        if start_epoch > 0 and checkpoint_path:
            resume_checkpoint = os.path.join(output_path, f"checkpoint-{start_epoch}")
            if not os.path.isfile(resume_checkpoint):
                raise FileNotFoundError(f"Resume checkpoint not found at: {resume_checkpoint}")
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")

        # Collate function
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

        training_args = self.get_training_args(output_path)
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            callbacks=[self.EpochEndCallback(self)],
        )

        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info("Training completed")

        #  Check if the model (checkpoint) has already completed training for the specified number of epochs, if so, can start again without resuming
        if 'start_epoch' in self.configuration and self.configuration['start_epoch'] == training_args.num_train_epochs:
            self.configuration['start_epoch'] = 0
            self.model_entity.update()


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
    model = project.models.get(model_name='rf-detr-sdk-clone-3')
    print("model done")
    model.status = 'pre-trained'
    model_adapter = HuggingAdapter(model)
    # model_adapter.configuration['start_epoch'] = 3
    # model_adapter.configuration['train_configs'] = {'num_train_epochs': 4}
    print("model_adapter done - start train")
    model_adapter.train_model(model=model)
    print("convert done")
