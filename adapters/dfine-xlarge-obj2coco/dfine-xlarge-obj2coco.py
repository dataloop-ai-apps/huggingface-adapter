import json
import os
from pathlib import Path
import PIL
import torch
import logging
import dtlpy as dl
from typing import List
from transformers import DFineForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
import shutil
from dtlpyconverters import services, coco_converters
from pycocotools.coco import COCO
from datasets import Dataset
from PIL import Image
from dotenv import load_dotenv

# installs :
# pip install datasets
# pip install git+https://github.com/dataloop-ai-apps/dtlpy-converters
# pip install -U transformers
# pip install python-dotenv


logger = logging.getLogger("[D-FINE]")


class HuggingAdapter(dl.BaseModelAdapter):
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

    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "d-fine")
        self.device = self.configuration.get("device", "cpu")
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if 'image/' not in item.mimetype:
            raise ValueError("Item must be an image for object detection.")
        return item

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
        print("need to implement save")

    def train(self, data_path: str, output_path: str, **kwargs) -> None:
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
        train_data = load_coco_as_list(
            os.path.join(data_path, "train", "_annotations.coco.json"), os.path.join(data_path, "train")
        )
        val_data = load_coco_as_list(
            os.path.join(data_path, "valid", "_annotations.coco.json"), os.path.join(data_path, "valid")
        )

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        # ------------------------- 1. preprocessing -------------------------
        def preprocess(example):
            image = Image.open(example["image"]).convert("RGB")

            # returns dict: {'pixel_values': tensor([1,3,H,W]), 'pixel_mask': ...}
            enc = self.processor(images=image, return_tensors="pt")
            enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove the batch dim

            # add label tensors in the format expected by the model
            enc["class_labels"] = torch.tensor(example["class_labels"], dtype=torch.long)
            enc["boxes"] = torch.tensor(example["boxes"], dtype=torch.float)

            return enc

        # Step 4: Map preprocessing
        train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)  # drop original cols
        val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

        # Tell the dataset to return torch tensors for the desired columns
        cols = ["pixel_values", "class_labels", "boxes"]
        train_dataset.set_format(type="torch", columns=cols)
        val_dataset.set_format(type="torch", columns=cols)

        # Step 5: Collate function
        # ------------------------- 3. custom collate_fn -------------------------
        def collate_fn(batch):
            pixel_values = torch.stack([b["pixel_values"] for b in batch])
            labels = [{"class_labels": b["class_labels"], "boxes": b["boxes"]} for b in batch]
            return {"pixel_values": pixel_values, "labels": labels}

        # Step 6: TrainingArguments and Trainer
        training_args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=1,
            num_train_epochs=4,
            logging_steps=10,
            save_strategy="epoch",
            #  evaluation_strategy="epoch",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
        )
        print("start real training")
        trainer.train()
        print("training done")


if __name__ == "__main__":
    print("start")
    use_rc_env = False
    if use_rc_env:
        dl.setenv('rc')
    else:
        dl.setenv('prod')

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
        # project = dl.projects.get(project_name='ShadiDemo')
        project = dl.projects.get(project_name='IPM development')
    print("project done")
    model = project.models.get(model_name='rd-dert-used-for-dfine-train-hfg')
    print("model done")
    model.status = 'pre-trained'
    model_adapter = HuggingAdapter(model)
    print("model_adapter done - start train")
    model_adapter.train_model(model=model)
    print("convert done")
