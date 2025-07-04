import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict, Any

import numpy as np
import PIL
import torch
from datasets import Dataset
from PIL import Image
from pycocotools.coco import COCO
import albumentations as A
from transformers import DFineForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import dtlpy as dl
from dtlpyconverters import services, coco_converters
from dtlpy.services import service_defaults


logger = logging.getLogger("[D-FINE]")


class HuggingAdapter(dl.BaseModelAdapter):

    class EpochEndCallback(TrainerCallback):
        """Custom callback for managing model checkpoints during training.

        This callback handles checkpoint management at the end of each epoch. By default, Hugging Face creates
        numbered checkpoints (e.g. checkpoint-1000, checkpoint-2000). This callback simplifies checkpoint
        management by:
        1. Keeping only 'best-checkpoint' - Copy of the checkpoint with best metrics
        2. Keeping only 'last-checkpoint' - Copy of the most recent checkpoint
        3. Removing all numbered checkpoints to save disk space

        Args:
            model_adapter_instance: Instance of the HuggingAdapter class
            faas_callback: Optional callback function for FaaS integration
        """

        def __init__(
            self, model_adapter_instance: 'HuggingAdapter', faas_callback: Optional[Callable[[int, int], None]] = None
        ) -> None:
            super().__init__()
            self.model_adapter = model_adapter_instance
            self.faas_callback = faas_callback

        def on_epoch_end(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
        ) -> TrainerControl:
            # state.epoch is a float where integer part = epoch number, decimal part = progress within epoch
            # Convert to int to get just the epoch number for logging and metrics
            current_epoch = int(state.epoch)
            total_epochs = int(args.num_train_epochs)
            logger.info(f"Epoch {current_epoch} ended")

            # FaaS callback
            if self.faas_callback:
                self.faas_callback(current_epoch, total_epochs)

            samples = []
            NaN_defaults = {'loss': 1.0, 'eval_loss': 1.0}
            # the only metrics that model is creating is loss and eval_loss
            loss_value = state.log_history[0]['loss'] if state.log_history else NaN_defaults['loss']
            eval_loss_value = (
                state.log_history[1]['eval_loss'] if len(state.log_history) > 1 else NaN_defaults['eval_loss']
            )
            print(f"-HHH- state.log_history {state.log_history}")
            logger.info(f"-HHH- state.log_history {state.log_history}")
            for metric_name, value in [('loss', loss_value), ('eval_loss', eval_loss_value)]:
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    logger.warning(f"Non-finite value for {metric_name}. Replacing with default.")
                    value = NaN_defaults.get(metric_name, 0)
                samples.append(dl.PlotSample(figure=metric_name, legend='metrics', x=current_epoch, y=value))

            if samples:
                self.model_adapter.model_entity.metrics.create(
                    samples=samples, dataset_id=self.model_adapter.model_entity.dataset_id
                )

            # Update internal configuration
            self.model_adapter.configuration['start_epoch'] = current_epoch + 1
            logger.info(f"best_model_checkpoint: {state.best_model_checkpoint}")
            if state.best_model_checkpoint:
                self.model_adapter.configuration['checkpoint_name'] = 'best-checkpoint'
            else:
                logger.info("No best model checkpoint available yet")
            self.model_adapter.model_entity.update()

            # Save model
            logger.info("Saving model checkpoint to model entity...")
            self.model_adapter.save_to_model(local_path=None, cleanup=False, replace=True, state=state)
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

    def collate_fn(self, batch):
        pixel_values = []
        labels = []

        for sample in batch:
            # Load image from disk as RGB numpy array
            np_img = np.array(Image.open(sample["image"]).convert("RGB"))

            bboxes = sample["objects"]["bbox"]
            categories = sample["objects"]["category"]
            # Apply Albumentations transform if available
            if sample['image_id'] in self.train_images_id:
                transformed = self.train_transform(image=np_img, bboxes=bboxes, category=categories)
            else:
                transformed = self.eval_transform(image=np_img, bboxes=bboxes, category=categories)
            np_img = transformed["image"]
            bboxes = transformed["bboxes"]
            categories = transformed["category"]

            annotations = []
            for category, bbox in zip(categories, bboxes):
                formatted_annotation = {
                    "image_id": sample["image_id"],
                    "category_id": category,
                    "bbox": list(bbox),
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
                annotations.append(formatted_annotation)

            formatted_annotations = {"image_id": sample["image_id"], "annotations": annotations}

            # Process the image with the processor
            processed = self.processor(images=np_img, annotations=formatted_annotations, return_tensors="pt")

            processed = {k: v[0] for k, v in processed.items()}

            pixel_values.append(processed["pixel_values"])
            labels.append(processed["labels"])
        # print(f"-HHH- labels: {labels}")
        # print(f"-HHH- pixel_values shape: {pixel_values[0].shape}")
        return {"pixel_values": torch.stack(pixel_values), "labels": labels}

    def _get_hugging_dataset(self, data_path: str) -> Tuple[Dataset, Dataset]:
        """Get Hugging Face datasets for training and validation.

        Args:
            data_path (str): Path to data directory containing train and valid subdirectories

        Returns:
            tuple[Dataset, Dataset]: Training and validation datasets
        """
        logger.info('get_hugging_dataset')

        def load_coco_as_list(subset: str) -> List[Dict[str, Any]]:
            # Calculate paths based on subset
            annotation_path = os.path.join(data_path, subset, "_annotations.coco.json")
            image_dir = os.path.join(data_path, subset)

            coco = COCO(annotation_path)
            items = []

            def fix_dimensions(bbox, width, height):
                # Unpack bbox coordinates
                x, y, w, h = bbox
                if x < 0:
                    x = 0
                if x + w > width:
                    w = width - x
                if y < 0:
                    y = 0
                if y + h > height:
                    h = height - y

                return [x, y, w, h]

            for image_id in coco.imgs:
                image_info = coco.loadImgs(image_id)[0]
                new_img_item = {
                    'image_id': image_id,
                    'image': os.path.join(image_dir, image_info['file_name']),
                    'width': image_info['width'],
                    'height': image_info['height'],
                }

                id_list = []
                area_list = []
                bbox_list = []
                category_list = []
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
                    id_list.append(ann["id"])
                    bbox_list.append(fix_dimensions(ann["bbox"], image_info['width'], image_info['height']))
                    category_list.append(ann["category_id"])
                    area_list.append(ann["area"])

                new_img_item['objects'] = {
                    "id": id_list,
                    "bbox": bbox_list,
                    "category": category_list,
                    "area": area_list,
                }
                items.append(new_img_item)
            return items

        # Step 2: Prepare datasets
        logger.info('load_coco_as_list train')
        train_dataset = Dataset.from_list(load_coco_as_list("train"))
        logger.info('load_coco_as_list val')
        val_dataset = Dataset.from_list(load_coco_as_list("valid"))

        logger.info('datasets map done')
        return train_dataset, val_dataset

    def save(self, local_path: str, **kwargs: Any) -> None:
        """Save model checkpoint to local path.

        Args:
            local_path: Path to model checkpoint directory
            **kwargs: Additional keyword arguments

        """
        if self.training_args is None or os.path.abspath(self.training_args.output_dir) == os.path.abspath(local_path):
            logger.error("Cannot save model - training args not initialized or output directory matches save path")
            return

        checkpoint_dir = self.training_args.output_dir
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        logger.info(f"save : checkpoints list : {checkpoints}")
        if not checkpoints:
            return

        # Sort checkpoints by number
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))

        # Copy latest checkpoint
        latest_checkpoint = checkpoints[-1]
        latest_src = os.path.join(checkpoint_dir, latest_checkpoint)
        latest_dst = os.path.join(local_path, 'last-checkpoint')
        if os.path.exists(latest_dst):
            shutil.rmtree(latest_dst)
        shutil.copytree(latest_src, latest_dst)

        # Copy best checkpoint if available
        state = kwargs.get('state')
        if state and state.best_model_checkpoint:
            best_checkpoint = os.path.basename(state.best_model_checkpoint)
            best_src = os.path.join(checkpoint_dir, best_checkpoint)
            if os.path.exists(best_src):
                best_dst = os.path.join(local_path, 'best-checkpoint')
                if os.path.exists(best_dst):
                    shutil.rmtree(best_dst)
                shutil.copytree(best_src, best_dst)
                self.configuration.update({'checkpoint_name': 'best-checkpoint'})

    def load(self, local_path: str, **kwargs: Any) -> None:
        """Load model from local path.

        Args:
            local_path: Path to model checkpoint directory
            **kwargs: Additional keyword arguments

        """
        logger.info(f"Loading model from {local_path}")

        # Get configuration parameters
        image_size = self.configuration.get("image_size", 640)
        self.model_name = self.configuration.get("model_name", "d-fine")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_processor_path = self.configuration.get("image_processor_path", "ustc-community/dfine-xlarge-obj2coco")
        checkpoint = self.configuration.get("checkpoint_name", "ustc-community/dfine-xlarge-obj2coco")
        checkpoint_path = os.path.join(local_path, checkpoint)

        # Initialize image processor
        self.processor = AutoImageProcessor.from_pretrained(
            "ustc-community/dfine-xlarge-obj2coco",
            do_resize=True,
            size={"width": image_size, "height": image_size},
            use_fast=True,
        )

        # Initialize transforms
        self.eval_transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_area=1, min_width=1, min_height=1),
        )
        augmentation_config = self.configuration.get('augmentation_config', {})
        logger.info(f"augmentation_config: {augmentation_config}")
        self.train_transform = A.Compose(
            [
                A.Rotate(limit=augmentation_config.get('rotate_limit', 15), p=augmentation_config.get('rotate_p', 0.5)),
                A.Perspective(p=augmentation_config.get('perspective_p', 0.1)),
                A.HorizontalFlip(p=augmentation_config.get('horizontal_flip_p', 0.5)),
                A.RandomBrightnessContrast(p=augmentation_config.get('brightness_contrast_p', 0.5)),
                A.HueSaturationValue(p=augmentation_config.get('hue_saturation_p', 0.1)),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_area=25, min_width=1, min_height=1),
        )

        # Prepare label mappings
        if 0 not in self.model_entity.dataset.instance_map.values():
            self.model_entity.dataset.instance_map = {
                label: label_id - 1 for label, label_id in self.model_entity.dataset.instance_map.items()
            }
        id2label = {label_id: label for label, label_id in self.model_entity.dataset.instance_map.items()}

        # Check if loading from checkpoint
        if checkpoint_path != "" and checkpoint_path.strip() != "":
            required_files = ['config.json', 'model.safetensors', 'training_args.bin', 'trainer_state.json']
            has_required_files = all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_files)
            if not has_required_files:
                logger.warning(f"Checkpoint path {checkpoint_path} does not contain required files: {required_files}")
            else:
                checkpoint = checkpoint_path

        # Load and initialize model
        self.model = DFineForObjectDetection.from_pretrained(
            checkpoint,
            num_labels=len(self.model_entity.dataset.instance_map),
            use_safetensors=True,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=self.model_entity.dataset.instance_map,
        )
        print(f"-HHH- id2label: {id2label}")
        print(f"-HHH- self.model_entity.dataset.instance_map: {self.model_entity.dataset.instance_map}")
        # self.model = DFineForObjectDetection.from_pretrained(
        #     checkpoint_name,
        #     num_labels=6,
        #     ignore_mismatched_sizes=True,
        #     id2label={0: "beetle", 1: "cockroach", 2: "fly", 3: "moth", 4: "other", 5: "small fly"},
        #     label2id={"beetle": 0, "cockroach": 1, "fly": 2, "moth": 3, "other": 4, "small fly": 5},
        # )
        # self.model.to(self.device)

        # Debug prints
        print(f"-HHH- self.model.config.id2label {self.model.config.id2label}")
        print(f"-HHH- self.model.config.label2id {self.model.config.label2id}")
        print(f"-HHH- self.model_entity.labels {self.model_entity.labels}")

    def predict(self, batch: List[np.ndarray], **kwargs: Any) -> List[dl.AnnotationCollection]:
        """Predict on a batch of images.

        Args:
            batch (List[np.ndarray]): List of images as numpy arrays
            **kwargs: Additional keyword arguments

        Returns:
            List[dl.AnnotationCollection]: List of annotations for each image
        """
        batch_annotations = []

        for item in batch:
            # Process numpy array image
            image = PIL.Image.fromarray(item).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            print(f"-HHH- target_sizes: {target_sizes}")

            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]

            # Create box annotations for each detection
            item_annotations = dl.AnnotationCollection()
            # Get top 5 detections by score
            scores = results["scores"].tolist()
            top_n = 5  # Number of top detections to keep
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            for idx in top_indices:
                score, label_id, box = results["scores"][idx], results["labels"][idx], results["boxes"][idx]

                score, label = score.item(), label_id.item()
                print(f"score: {score}, label: {label}, box: {box}")

                if score < 0.29:
                    print("skipping")
                    continue
                box = [round(i, 2) for i in box.tolist()]
                left, top, right, bottom = box
                # Create box annotation
                print(f"-HHH- top: {top}, left: {left}, bottom: {bottom}, right: {right}")
                item_annotations.add(
                    annotation_definition=dl.Box(
                        label=self.model.config.id2label[label], top=top, left=left, bottom=bottom, right=right
                    ),
                    model_info={"name": self.model_name, "model_id": self.model_name, "confidence": round(score, 3)},
                )
            batch_annotations.append(item_annotations)

        return batch_annotations

    def convert_from_dtlpy(self, data_path: str, **kwargs: Any) -> None:
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
        """Get training arguments for the model.

        This method creates and returns a TrainingArguments object with training hyperparameters
        configured based on the model's configuration dictionary.

        Args:
            output_path (str): Directory path where model outputs and checkpoints will be saved

        Returns:
            TrainingArguments: Hugging Face TrainingArguments object configured with training parameters
                including batch sizes, learning rate, epochs, logging/saving strategies etc.
        """
        cfg = self.configuration.get('train_configs', {})
        logger.info(f'train_config_dict: {cfg}')

        return TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=cfg.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=cfg.get('per_device_eval_batch_size', 8),
            gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
            learning_rate=cfg.get('learning_rate', 5e-5),
            weight_decay=cfg.get('weight_decay', 0.0),
            num_train_epochs=cfg.get('num_train_epochs', 3),
            warmup_steps=cfg.get('warmup_steps', 300),
            max_grad_norm=cfg.get('max_grad_norm', 0.1),
            logging_strategy="epoch",
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=cfg.get('metric_for_best_model', "loss"),
            greater_is_better=cfg.get('greater_is_better', False),
            resume_from_checkpoint=cfg.get('resume_from_checkpoint', None),
            remove_unused_columns=False,
            fp16=cfg.get('fp16', False),
            disable_tqdm=True,  # Disable progress bars
        )

    def train(self, data_path: str, output_path: str, **kwargs: Any) -> None:
        """Train the model on the provided dataset.

        This method handles the complete training process including:
        1. Loading and preparing training/validation datasets
        2. Setting up checkpointing and resuming from previous checkpoints
        3. Configuring the training arguments and trainer
        4. Running the training loop with evaluation
        5. Managing training state and model checkpoints

        Args:
            data_path (str): Path to directory containing training data in COCO format
            output_path (str): Directory path where model outputs and checkpoints will be saved
            **kwargs: Additional keyword arguments (unused)

        Returns:
            None

        Raises:
            FileNotFoundError: If resume checkpoint is not found when resuming training
        """
        train_dataset, val_dataset = self._get_hugging_dataset(data_path)
        self.train_images_id = [sample['image_id'] for sample in train_dataset]

        print(f"Example train sample: {train_dataset[0]}")
        print(f"Example val sample: {val_dataset[0]}")

        # Resume from checkpoint logic
        start_epoch = self.configuration.get('start_epoch', 1)

        resume_checkpoint = None
        if start_epoch > 1:
            resume_checkpoint = os.path.join(
                service_defaults.DATALOOP_PATH, "models", self.model_entity.name, 'last-checkpoint'
            )
            if not os.path.isdir(resume_checkpoint):
                raise FileNotFoundError(f"Resume checkpoint not found at: {resume_checkpoint}")
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            resume_checkpoint = os.path.abspath(resume_checkpoint)

        self.training_args = self.get_training_args(output_path)
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            data_collator=lambda batch: self.collate_fn(batch),
            callbacks=[self.EpochEndCallback(self)],
        )

        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        # trainer.train()
        logger.info("Training completed")

        #  Check if the model (checkpoint) has already completed training for the specified number of epochs, if so, can start again without resuming
        start_epoch = self.configuration.get('start_epoch', 1)
        if start_epoch == self.training_args.num_train_epochs + 1:
            self.configuration['start_epoch'] = 1
            self.model_entity.update()

    def embed(self, batch: List[dl.Item], **kwargs: Any) -> None:
        """Embed items - not implemented for this adapter."""
        raise NotImplementedError("Embed method not implemented for this adapter")


if __name__ == "__main__":
    print("start")
    use_rc_env = False
    if use_rc_env:
        dl.setenv('rc')
    else:
        dl.setenv('prod')
    # from dotenv import load_dotenv

    # load_dotenv()
    # api_key = os.getenv('DTLPY_API_KEY')
    # print(f"api_key: {api_key}")
    # if api_key:
    #     print("DTLPY_API_KEY found in environment variables")
    #     dl.login_api_key(api_key=api_key)
    # else:
    #     print("ERROR: DTLPY_API_KEY not found in environment variables")
    #     raise ValueError("Missing required DTLPY_API_KEY environment variable")

    # if dl.token_expired():
    #     dl.login()
    # print("login done")
    if use_rc_env:
        project = dl.projects.get(project_name='Husam Testing')
    else:
        # project = dl.projects.get(project_name='ShadiDemo')
        project = dl.projects.get(project_name='IPM development')
    print("project done")
    # model = project.models.get(model_name='dfine-sdk-helios-1-4')
    model = project.models.get(model_name='dfine-sdk-rodents-full-4')
    print("model done")
    model.status = 'pre-trained'
    model_adapter = HuggingAdapter(model)
    model_adapter.configuration['start_epoch'] = 1
    model_adapter.configuration['checkpoint_name'] = "ustc-community/dfine-xlarge-obj2coco"
    model_adapter.configuration['train_configs'] = {
        'num_train_epochs': 1,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 8,
    }
    # print("run predict")
    model_adapter.train_model(model=model)
    # _, annotations = model_adapter.predict_items(items=[project.items.get(item_id='68663838d1c962555c310091')])
# print(annotations)
# model_adapter.configuration['start_epoch'] = 4
# # model_adapter.configuration['checkpoint_name'] = 'best-checkpoint'
# model_adapter.configuration['train_configs'] = {'num_train_epochs': 7}
# print("model_adapter done - start train")
# model_adapter.train_model(model=model)
# print("convert done")
