
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor
from hugging_base import HuggingBase

class HuggingAdapter(HuggingBase):
    def load_model_and_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")

    def get_cmd(self):
        return "python script.py"

    def predict(self, item):
        buffer = item.download(save_locally=False)
        image = Image.open(buffer)

        inputs = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

        predictions = []
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                predictions.append(f"{self.model.config.id2label[label]}: {score:.2f} {box}")
        
        return predictions
