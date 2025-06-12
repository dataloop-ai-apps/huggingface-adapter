
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

    def predict(self, image_url):
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)
        return results

# Example usage
if __name__ == "__main__":
    adapter = HuggingAdapter()
    adapter.load_model_and_processor()
    print(adapter.get_cmd())
