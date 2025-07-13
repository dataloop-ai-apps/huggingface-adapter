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
        """Callback class that handles end-of-epoch events during model training.

        This callback is triggered at the end of each training epoch to:
        - Log epoch completion and metrics
        - Call any provided FaaS callback function
        - Record training metrics (loss, eval_loss) in Dataloop
        - Update model configuration with current epoch and best checkpoint

        Attributes:
            model_adapter (HuggingAdapter): Reference to parent model adapter instance
            faas_callback (Optional[Callable[[int, int], None]]): Optional callback function that takes
                current_epoch and total_epochs as arguments
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
            current_epoch = int(state.epoch) if state.epoch is not None else 0
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
            logger.info(f"end of epoch {current_epoch} state.log_history {state.log_history}")
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
            self.model_adapter.model_entity.update()

            # Save model
            logger.info("Saving model checkpoint to model entity...")
            self.model_adapter.save_to_model(local_path=None, cleanup=False, replace=True, state=state)
            return control

    @staticmethod
    def _process_coco_json(output_annotations_path: str) -> None:
        """
        Process COCO JSON annotations file to make it compatible with D-FINE requirements.

        D-FINE requires integer IDs and supercategory fields for categories. This function converts
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
        """Collate function for batching data during training.

        This function processes a batch of samples and prepares them for the model by:
        1. Loading images from disk and converting to RGB numpy arrays
        2. Applying data augmentation transforms (training) or basic transforms (validation)
        3. Formatting bounding boxes and annotations in COCO format
        4. Processing images and annotations through the image processor
        5. Stacking pixel values and labels for batch processing

        Args:
            batch: List of sample dictionaries containing:
                - image (str): Path to image file
                - image_id (int): Unique identifier for the image
                - objects (dict): Dictionary containing:
                    - bbox (List[List[float]]): List of bounding boxes in [x, y, width, height] format
                    - category (List[int]): List of category IDs for each bounding box

        Returns:
            dict: Dictionary containing:
                - pixel_values (torch.Tensor): Stacked processed image tensors
                - labels (List[dict]): List of annotation dictionaries for each sample
        """
        pixel_values = []
        labels = []

        for sample in batch:
            # Load image from disk as RGB numpy array
            np_img = np.array(Image.open(sample["image"]).convert("RGB"))

            bboxes = sample["objects"]["bbox"]
            categories = sample["objects"]["category"]
            # Training data gets augmentations like random flips/crops for better generalization
            # Validation data only gets basic resizing to keep evaluation consistent
            if sample['image_id'] in self.train_images_id:
                transformed = self.train_transform(image=np_img, bboxes=bboxes, category=categories)
            else:
                transformed = self.eval_transform(image=np_img, bboxes=bboxes, category=categories)
            np_img = transformed["image"]
            bboxes = transformed["bboxes"]
            categories = transformed["category"]

            annotations_list = []
            for category, bbox in zip(categories, bboxes):
                formatted_annotation = {
                    "image_id": sample["image_id"],
                    "category_id": category,
                    "bbox": list(bbox),
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
                annotations_list.append(formatted_annotation)

            formatted_annotations = {"image_id": sample["image_id"], "annotations": annotations_list}

            # Process the image with the processor
            processed = self.processor(images=np_img, annotations=formatted_annotations, return_tensors="pt")

            processed = {k: v[0] for k, v in processed.items()}

            pixel_values.append(processed["pixel_values"])
            labels.append(processed["labels"])
        return {"pixel_values": torch.stack(pixel_values), "labels": labels}

    def _get_hugging_dataset(self, data_path: str) -> Tuple[Dataset, Dataset]:
        """Get Hugging Face datasets for training and validation.

        This method loads and prepares COCO format datasets for D-FINE model training.
        It processes COCO JSON annotations and image files to create Hugging Face
        datasets with proper formatting for object detection training.

        Args:
            data_path (str): Path to directory containing training data in COCO format.
                           Expected structure: data_path/train/ and data_path/valid/
                           with _annotations.coco.json files in each subdirectory.

        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets formatted for
                                   Hugging Face training with image paths, bounding boxes,
                                   and category information.

        Note:
            The method automatically fixes bounding box dimensions to ensure they
            stay within image boundaries and handles COCO format annotations.
        """
        logger.info('get_hugging_dataset')

        def load_coco_as_list(subset: str) -> List[Dict[str, Any]]:
            # Calculate paths based on subset
            annotation_path = os.path.join(data_path, subset, "_annotations.coco.json")
            image_dir = os.path.join(data_path, subset)

            coco = COCO(annotation_path)
            items = []

            # Ensure bboxes are within image bounds
            def fix_dimensions(bbox, width, height):
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

        This method saves the latest and best model checkpoints from the training output
        directory to the specified local path. It handles both the most recent checkpoint
        and the best performing checkpoint based on evaluation metrics.

        Args:
            local_path (str): Directory path where model checkpoint will be saved
            **kwargs: Additional keyword arguments including training state

        Note:
            - Copies the latest checkpoint to 'last-checkpoint' directory
            - Copies the best checkpoint (if available) to 'best-checkpoint' directory
            - Updates configuration to use 'best-checkpoint' if it exists
            - Skips saving if training args are not initialized or output directory
              matches the save path
        """

        logger.info(f"HuggingAdapter.save local_path: {local_path}")

        # Print subdirectories of local_path
        if os.path.exists(local_path):
            subdirs = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
            if len(subdirs) != 0:
                logger.info(f"HuggingAdapter.save subdirs: {subdirs}")

        # save only to tmp dir, ignore calles from base model
        if (
            self.training_args is None
            or self.training_args.output_dir is None
            or os.path.abspath(self.training_args.output_dir) == os.path.abspath(local_path)
        ):
            logger.error("Cannot save model - training args not initialized or output directory matches save path")
            return

        checkpoint_dir = self.training_args.output_dir
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        logger.info(f"HuggingAdapter.save checkpoints list : {checkpoints}")
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
        logger.info(f"HuggingAdapter.save copying latest checkpoint from {latest_src} to {latest_dst}")
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
                logger.info(f"HuggingAdapter.save copying best checkpoint from {best_src} to {best_dst}")
                shutil.copytree(best_src, best_dst)
                logger.info(f"HuggingAdapter.save updating configuration with best checkpoint")
                self.configuration.update({'checkpoint_name': 'best-checkpoint'})

    def load(self, local_path: str, **kwargs: Any) -> None:
        """Load model from local path.

        This method initializes the D-FINE model for object detection by:
        1. Loading configuration parameters (image size, model name, device, etc.)
        2. Initializing the image processor with specified settings
        3. Setting up data augmentation transforms for training and evaluation
        4. Loading the pre-trained model with proper label mappings
        5. Moving the model to the appropriate device (CPU/GPU)

        Args:
            local_path (str): Path to model checkpoint directory
            **kwargs: Additional keyword arguments (unused)

        Note:
            The method automatically handles label ID mapping to ensure compatibility
            with D-FINE requirements (IDs starting from 0).
        """
        logger.info(f"Loading model from {local_path}")

        # Get configuration parameters
        image_size = self.configuration.get("image_size", 640)
        self.model_name = self.configuration.get("model_name", "d-fine")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = self.configuration.get("confidence_threshold", 0.25)
        image_processor_path = self.configuration.get("image_processor_path", "ustc-community/dfine-xlarge-obj2coco")
        checkpoint = self.configuration.get("checkpoint_name", "ustc-community/dfine-xlarge-obj2coco")
        checkpoint_path = os.path.join(local_path, checkpoint)
        logger.info(
            f"Model Configuration:\n"
            f"Image Size: {image_size}\n"
            f"Model Name: {self.model_name}\n"
            f"Device: {self.device}\n"
            f"Confidence Threshold: {self.confidence_threshold}\n"
            f"Image Processor Path: {image_processor_path}\n"
            f"Checkpoint: {checkpoint}\n"
            f"Checkpoint Path: {checkpoint_path}"
        )

        # Initialize image processor
        self.processor = AutoImageProcessor.from_pretrained(
            image_processor_path, do_resize=True, size={"width": image_size, "height": image_size}, use_fast=True
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

        # Prepare label mappings , dfine wants the ids to start from 0
        if 0 not in self.model_entity.dataset.instance_map.values():
            self.model_entity.dataset.instance_map = {
                label: label_id - 1 for label, label_id in self.model_entity.dataset.instance_map.items()
            }
        id2label = {label_id: label for label, label_id in self.model_entity.dataset.instance_map.items()}

        # Print subdirectories of checkpoint_path if it exists
        if os.path.exists(checkpoint_path):
            subdirs = [d for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
            logger.info(f"Subdirectories in {checkpoint_path}:")
            for subdir in subdirs:
                logger.info(f"  - {subdir}")

        # Check if loading from checkpoint
        if checkpoint_path != "" and checkpoint_path.strip() != "":
            required_files = ['config.json', 'model.safetensors', 'training_args.bin', 'trainer_state.json']
            has_required_files = all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_files)
            if not has_required_files:
                logger.warning(f"Checkpoint path {checkpoint_path} does not contain required files: {required_files}")
            else:
                checkpoint = checkpoint_path

        # Load and initialize model
        logger.info(f"HuggingAdapter.load checkpoint: {checkpoint}")
        self.model = DFineForObjectDetection.from_pretrained(
            checkpoint,
            num_labels=len(self.model_entity.dataset.instance_map),
            use_safetensors=True,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=self.model_entity.dataset.instance_map,
        )

        # Move model to the correct device
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        self.pooling_method = self.configuration.get("pooling_method", "max")
        logger.info(f"Using pooling method for embeddings: {self.pooling_method}")

    def predict(self, batch: List[np.ndarray], **kwargs: Any) -> List[dl.AnnotationCollection]:
        """Predict on a batch of images.

        This method performs object detection on a batch of images using the loaded D-FINE model.
        The prediction process includes:
        1. Converting numpy arrays to PIL Images
        2. Processing images through the image processor
        3. Running inference with the model
        4. Post-processing detection results
        5. Filtering detections by confidence threshold
        6. Creating Dataloop annotation collections

        Args:
            batch (List[np.ndarray]): List of images as numpy arrays with shape (H, W, C)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            List[dl.AnnotationCollection]: List of annotation collections for each image,
                                         containing bounding box annotations with labels,
                                         confidence scores, and model information.

        Note:
            Only detections with confidence scores above the configured threshold
            (default: 0.25) are included in the results.
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
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)

            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

            # Create box annotations for each detection
            item_annotations = dl.AnnotationCollection()

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score, label = score.item(), label.item()
                logger.info(f"score: {score}, label: {label}, box: {box}")
                if score < self.confidence_threshold:
                    logger.info(f"score: {score} < {self.confidence_threshold}, skipping")
                    continue
                box = [round(i, 2) for i in box.tolist()]
                left, top, right, bottom = box
                # Create box annotation
                item_annotations.add(
                    annotation_definition=dl.Box(
                        label=self.model.config.id2label[label], top=top, left=left, bottom=bottom, right=right
                    ),
                    model_info={"name": self.model_name, "model_id": self.model_name, "confidence": round(score, 3)},
                )
            batch_annotations.append(item_annotations)

        return batch_annotations

    def convert_from_dtlpy(self, data_path: str, **kwargs: Any) -> None:
        """Convert dataset from Dataloop format to COCO format for D-FINE training.

        This method converts a Dataloop dataset to the COCO format required by D-FINE model training.
        The conversion process includes:
        1. Converting Dataloop annotations to COCO format for each subset (train/validation)
        2. Processing COCO JSON files to ensure compatibility with D-FINE requirements
        3. Reorganizing image files from nested directory structure to flat structure
        4. Handling validation subset name conversion (validation -> valid)
        5. Moving processed files to the correct directory structure

        Args:
            data_path (str): Path to directory where converted data will be stored
            **kwargs: Additional keyword arguments (unused)

        Raises:
            ValueError: If model has no labels defined
            FileNotFoundError: If no image files are found in the source directory
            Exception: If conversion process fails for any subset
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
            metric_for_best_model=cfg.get('metric_for_best_model', "eval_loss"),
            greater_is_better=cfg.get('greater_is_better', False),
            resume_from_checkpoint=cfg.get('resume_from_checkpoint', None),
            remove_unused_columns=False,
            fp16=cfg.get('fp16', False),
            disable_tqdm=True,  # Disable progress bars
            eval_do_concat_batches=False,
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
        self.train_images_id = [sample['image_id'] for sample in train_dataset]  # type: ignore

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

        # Clean up all checkpoints from checkpoint_dir since we've copied what we need
        checkpoints = [d for d in os.listdir(self.training_args.output_dir) if d.startswith('checkpoint-')]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(self.training_args.output_dir, checkpoint)
            logger.info(f"HuggingAdapter.train Removing checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)

    def embed(self, batch, **kwargs):
        """Model feature vectors (embedding) on batch of images

        There are four pooling methods:
        - Mean pooling: Images with similar overall content will cluster together
        - Max pooling: Images with similar prominent objects will cluster together
        - CLS pooling: Images with similar scene types will cluster together
        - Attention pooling: Images with similar "important" objects will cluster together

        :param batch: `np.ndarray` - batch of images
        :return: `list[np.ndarray]` per each image / item in the batch
        """
        # Convert numpy arrays to PIL images
        images = [Image.fromarray(img) for img in batch]

        # Process through D-FINE model
        inputs = self.processor(images=images, return_tensors="pt")

        with torch.no_grad():
            # Get model outputs with hidden states
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get the last hidden state (decoder output)
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch, num_queries, hidden_size]

            # Apply pooling to get a fixed-size representation
            if self.pooling_method == 'mean':
                # Average over all object queries
                embeddings = torch.mean(last_hidden_state, dim=1)
            elif self.pooling_method == 'max':
                # Max pooling over object queries
                embeddings = torch.max(last_hidden_state, dim=1)[0]
            elif self.pooling_method == 'cls':
                # Use first query as "CLS" token
                embeddings = last_hidden_state[:, 0, :]
            elif self.pooling_method == 'attention':
                # Attention-weighted pooling
                attention_weights = F.softmax(torch.sum(last_hidden_state, dim=-1), dim=-1)
                embeddings = torch.sum(last_hidden_state * attention_weights.unsqueeze(-1), dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")

            embeddings = embeddings.cpu().detach().numpy().tolist()

        # Return list of numpy arrays (one per image)
        return embeddings
