
import torch
import requests
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
from hugging_base import HuggingBase

class HuggingAdapter(HuggingBase):
    def load_model_and_processor(self):
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")

    def get_cmd(self):
        return "python script.py --image <image_url>"

    def predict(self, input_items):
        results = []
        for item in input_items:
            image = Image.open(item.download(save_locally=False))
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self.processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)
            formatted_result = []
            for res in result:
                for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"]):
                    score, label = score.item(), label_id.item()
                    box = [round(i, 2) for i in box.tolist()]
                    formatted_result.append(f"{self.model.config.id2label[label]}: {score:.2f} {box}")
            results.append(formatted_result)
        return results

# Example usage
if __name__ == "__main__":
    adapter = HuggingAdapter()
    adapter.load_model_and_processor()
    print(adapter.get_cmd())
