
from transformers import AutoImageProcessor, DFineForObjectDetection
from hugging_base import HuggingBase

class HuggingAdapter(HuggingBase):
    def load_model_and_processor(self):
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
        self.model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")

    def get_cmd(self):
        return "python script.py --image <image_url>"

# Example usage
if __name__ == "__main__":
    adapter = HuggingAdapter()
    adapter.load_model_and_processor()
    print(adapter.get_cmd())
