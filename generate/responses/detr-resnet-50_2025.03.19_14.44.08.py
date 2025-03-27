
from transformers import DetrImageProcessor, DetrForObjectDetection
from hugging_base import HuggingBase

class HuggingAdapter(HuggingBase):
    def load_model_and_processor(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def get_cmd(self):
        return "python script.py --image <image_url>"

# Example usage
if __name__ == "__main__":
    adapter = HuggingAdapter()
    adapter.load_model_and_processor()
    print(adapter.get_cmd())
